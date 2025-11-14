import numpy as np
from scipy.stats import beta, bernoulli
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import glob
import re
import xarray as xr # If used in your ARMultiAnnDataset
import multiprocessing as mp
from scipy import ndimage # For connected components in placeholder shape prior
from typing   import List, Tuple


FNAME_RE = re.compile(r"data-(\d{4}-\d{2}-\d{2})-.*?_(\d+)\.nc$")

class ARMultiAnnDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, var_names: List[str]):
        super().__init__()
        self.root, self.var_names = Path(root), var_names

        by_date = {}
        for f in glob.glob(str(self.root / "*.nc")):
            m = FNAME_RE.search(Path(f).name)
            if m:
                by_date.setdefault(m.group(1), {})[int(m.group(2))] = Path(f)

        self.samples = sorted(by_date.items())
        self.real_max_ann = max(len(d) for _,d in self.samples)

    def _load_nc(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = xr.open_dataset(path, engine="netcdf4")

        # squeeze() drops ANY singleton dimension, e.g. the leading "time=1"
        vars_arr = [
            torch.as_tensor(ds[v].values).squeeze()     # (H, W) after squeeze
            for v in self.var_names
        ]
        x = torch.stack(vars_arr).float()               # [C, H, W]
        
        raw = torch.as_tensor(ds["LABELS"].values).squeeze()
        y   = (raw == 2).float().unsqueeze(0) 

        # y = torch.as_tensor(ds["LABELS"].values).squeeze().unsqueeze(0).float()
        ds.close()
        return x, y

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        _, ann_map = self.samples[idx]
        x_vars, y_list = None, []
        for p in ann_map.values():
            x, y = self._load_nc(p)
            if x_vars is None: x_vars = x
            y_list.append(y)
        y_cons = torch.stack(y_list).mean(0)
        return x_vars, y_list, y_cons      # variable-length list

def collate_var(batch):
    xs, y_lists, y_cons = zip(*batch)
    return torch.stack(xs), list(y_lists), torch.stack(y_cons)



# --- MCMC Helper functions (BETA_PRIORS, LAMBDA_SMOOTH, update_rater_performance, etc.) ---
# (These are mostly the same, but calculate_log_prob_L_pixel is now LOCAL)
BETA_PRIOR_SE_A, BETA_PRIOR_SE_B = 2, 2
BETA_PRIOR_SP_A, BETA_PRIOR_SP_B = 2, 2
LAMBDA_SMOOTH = 1.0 # Strength for local smoothness prior

# New Hyperparameter for the global shape prior
LAMBDA_SHAPE = 0.5 # Adjust based on the scale of your shape energy/log-prior

def update_rater_performance(L_current, rater_mask_k):
    # (Identical to previous version)
    N_11 = np.sum((L_current == 1) & (rater_mask_k == 1))
    N_10 = np.sum((L_current == 1) & (rater_mask_k == 0))
    N_00 = np.sum((L_current == 0) & (rater_mask_k == 0))
    N_01 = np.sum((L_current == 0) & (rater_mask_k == 1))
    se_k_posterior_a = BETA_PRIOR_SE_A + N_11
    se_k_posterior_b = BETA_PRIOR_SE_B + N_10
    new_se_k = np.clip(beta.rvs(se_k_posterior_a, se_k_posterior_b), 0.01, 0.99)
    sp_k_posterior_a = BETA_PRIOR_SP_A + N_00
    sp_k_posterior_b = BETA_PRIOR_SP_B + N_01
    new_sp_k = np.clip(beta.rvs(sp_k_posterior_a, sp_k_posterior_b), 0.01, 0.99)
    return new_se_k, new_sp_k

def get_neighbors(r, c, img_dims, connectivity=8):
    # (Updated version for 8-connectivity)
    neighbors = []
    if r > 0: neighbors.append((r - 1, c))
    if r < img_dims[0] - 1: neighbors.append((r + 1, c))
    if c > 0: neighbors.append((r, c - 1))
    if c < img_dims[1] - 1: neighbors.append((r, c + 1))
    if connectivity == 8:
        if r > 0 and c > 0: neighbors.append((r - 1, c - 1))
        if r > 0 and c < img_dims[1] - 1: neighbors.append((r - 1, c + 1))
        if r < img_dims[0] - 1 and c > 0: neighbors.append((r + 1, c - 1))
        if r < img_dims[0] - 1 and c < img_dims[1] - 1: neighbors.append((r + 1, c + 1))
    return neighbors

def calculate_log_prob_L_pixel_local(r, c, L_val, L_current, rater_masks_all,
                                     se_raters, sp_raters, img_dims):
    """
    Calculates log-probability for a pixel flip based ONLY on
    local likelihood (rater agreement) and local smoothness prior.
    The global shape prior is NOT included here.
    """
    log_p = 0.0
    eps = 1e-9
    # Likelihood part
    for k in range(len(rater_masks_all)):
        R_kij = rater_masks_all[k][r, c]
        se_k = se_raters[k]
        sp_k = sp_raters[k]
        if L_val == 1:
            if R_kij == 1: log_p += np.log(se_k if se_k > eps else eps)
            else: log_p += np.log((1 - se_k) if (1 - se_k) > eps else eps)
        else: # L_val == 0
            if R_kij == 0: log_p += np.log(sp_k if sp_k > eps else eps)
            else: log_p += np.log((1 - sp_k) if (1 - sp_k) > eps else eps)

    # Local smoothness prior part
    smoothness_penalty = 0
    neighbors = get_neighbors(r, c, img_dims, connectivity=8) # Using 8-connectivity
    for nr, nc in neighbors:
        if L_val != L_current[nr, nc]:
            smoothness_penalty += 1
    log_p += -LAMBDA_SMOOTH * smoothness_penalty
    return log_p

def calculate_global_shape_log_prior(L_mask, lambda_shape_param):
    """
    Placeholder for calculating the log of the global shape prior.
    This is where your SDF analysis or other global metrics would go.
    L_mask: The full 2D binary mask.
    lambda_shape_param: Strength of this prior.

    For demonstration, a VERY simple placeholder:
    - Penalize if there is not exactly one connected component.
    - Penalize if the AR is "too small" or "too large".
    A real implementation would use SDF properties or sophisticated AR metrics.
    """
    eps = 1e-9
    if np.sum(L_mask) < 10 : # If mask is essentially empty, no shape penalty/bonus
        return 0.0

    labeled_mask, num_features = ndimage.label(L_mask)
    log_prior_shape = 0.0

    # 1. Penalize if not roughly one main AR feature
    # This is a very crude way to encourage a single, coherent AR.
    # A more advanced method would analyze the properties of the largest component.
    if num_features == 0 :
        log_prior_shape -= 10 # Penalize empty mask if sum was > 0 but no features (should not happen)
    elif num_features > 2: # Penalize if it's too fragmented
        log_prior_shape -= (num_features - 1) * 2.0 # Penalty increases with fragmentation

    # 2. Placeholder: Basic aspect ratio of the largest component
    if num_features > 0:
        largest_component_size = 0
        largest_component_label = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_mask == i)
            if size > largest_component_size:
                largest_component_size = size
                largest_component_label = i
        
        if largest_component_label > 0:
            component_mask = (labeled_mask == largest_component_label)
            rows, cols = np.where(component_mask)
            if rows.size > 5: # Minimum size for aspect ratio calculation
                height = rows.max() - rows.min() + 1
                width = cols.max() - cols.min() + 1
                aspect_ratio = max(height, width) / (min(height, width) + eps) # eps to avoid div by zero
                
                # Reward elongation, penalize "square-ish" or non-elongated
                # Ideal AR is long and narrow, so high aspect ratio is good.
                if aspect_ratio < 2.0: # Penalize if not at least 2:1
                    log_prior_shape -= (2.0 - aspect_ratio) * 1.0
                elif aspect_ratio > 1.5: # Slight reward for being somewhat elongated
                    log_prior_shape += (aspect_ratio - 1.5) * 0.5


    # The returned value should be log(P_shape(L_mask))
    # which is often -lambda_shape * Energy_shape(L_mask)
    return lambda_shape_param * log_prior_shape # Note: lambda_shape_param scales the effect.
                                            # If energy is defined as something to minimize,
                                            # then it's -lambda_shape_param * energy


def run_mcmc_for_image(rater_masks_all, img_dims, n_raters,
                       L_initial_guess=None,
                       n_global_steps=100, # Total global MCMC iterations
                       n_gibbs_sweeps_local=5, # Gibbs sweeps for local refinement
                       n_burn_in_global=20,  # Burn-in for the global MCMC
                       image_idx_for_print="N/A"):
    """
    Main MCMC loop for a single image, incorporating global shape prior
    via Metropolis-Hastings after local Gibbs sweeps.
    """
    # Initialize L
    if L_initial_guess is not None and L_initial_guess.shape == img_dims:
        L_current = L_initial_guess.copy().astype(int)
    else:
        L_current = np.random.randint(0, 2, size=img_dims) # Random start if no/bad initial

    # Initialize rater parameters
    se_raters = [beta.rvs(BETA_PRIOR_SE_A, BETA_PRIOR_SE_B) for _ in range(n_raters)]
    sp_raters = [beta.rvs(BETA_PRIOR_SP_A, BETA_PRIOR_SP_B) for _ in range(n_raters)]

    L_samples = []
    se_samples = [[] for _ in range(n_raters)]
    sp_samples = [[] for _ in range(n_raters)]

    # Calculate initial global shape prior for L_current
    log_global_prior_L_current = calculate_global_shape_log_prior(L_current, LAMBDA_SHAPE)

    print_freq_global = n_global_steps // 10 if n_global_steps >=10 else 1

    for g_step in range(n_global_steps):
        if g_step > 0 and g_step % print_freq_global == 0:
             print(f"    (Img {image_idx_for_print}, Proc {mp.current_process().pid if 'mp' in globals() else 'main'}) Global Step {g_step}/{n_global_steps}")

        # 1. Update Rater Performance
        for k_rater in range(n_raters):
            se_raters[k_rater], sp_raters[k_rater] = update_rater_performance(L_current, rater_masks_all[k_rater])

        # 2. Propose L_candidate via Local Gibbs Sweeps
        L_candidate = L_current.copy()
        for _ in range(n_gibbs_sweeps_local): # Local refinement sweeps
            pixel_indices = [(r, c) for r in range(img_dims[0]) for c in range(img_dims[1])]
            np.random.shuffle(pixel_indices)
            for r_pix, c_pix in pixel_indices:
                log_p_L_is_1_local = calculate_log_prob_L_pixel_local(r_pix, c_pix, 1, L_candidate, rater_masks_all, se_raters, sp_raters, img_dims)
                log_p_L_is_0_local = calculate_log_prob_L_pixel_local(r_pix, c_pix, 0, L_candidate, rater_masks_all, se_raters, sp_raters, img_dims)
                
                max_log_p = max(log_p_L_is_1_local, log_p_L_is_0_local)
                if np.isneginf(max_log_p): prob_L_is_1_normalized = 0.5
                else:
                    p_L_is_1 = np.exp(log_p_L_is_1_local - max_log_p)
                    p_L_is_0 = np.exp(log_p_L_is_0_local - max_log_p)
                    sum_p = p_L_is_1 + p_L_is_0
                    if sum_p > 0: prob_L_is_1_normalized = p_L_is_1 / sum_p
                    else: prob_L_is_1_normalized = 0.5
                L_candidate[r_pix, c_pix] = bernoulli.rvs(prob_L_is_1_normalized)
        
        # 3. Metropolis-Hastings step for Global Shape Prior
        # The proposal L_candidate was generated using P_local(L|Data,Theta).
        # We accept it based on the P_global_shape.
        log_global_prior_L_candidate = calculate_global_shape_log_prior(L_candidate, LAMBDA_SHAPE)
        
        # Log Acceptance Ratio: log(P_global(L_cand)) - log(P_global(L_curr))
        # This implicitly assumes the proposal probability related to P_local roughly cancels or
        # that this is a Metropolis adjustment for the part of the prior not included in Gibbs.
        log_acceptance_ratio = log_global_prior_L_candidate - log_global_prior_L_current

        if np.log(np.random.rand()) < log_acceptance_ratio:
            L_current = L_candidate.copy()
            log_global_prior_L_current = log_global_prior_L_candidate # Update current global prior value
            # print(f"        (Img {image_idx_for_print}) Global proposal accepted.") # Can be too verbose

        # 4. Store Samples (after global burn-in)
        if g_step >= n_burn_in_global:
            L_samples.append(L_current.copy())
            for k_rater in range(n_raters):
                se_samples[k_rater].append(se_raters[k_rater])
                sp_samples[k_rater].append(sp_raters[k_rater])
                
    return L_samples, se_samples, sp_samples

# --- Worker Function for Multiprocessing (Calls the new MCMC loop) ---
def process_image_worker(args):
    image_idx, x_vars_tensor, y_list_tensors, y_cons_tensor, \
    n_global_mcmc_steps, n_gibbs_sweeps_local, n_global_burn_in = args
    
    worker_pid = mp.current_process().pid
    # Print progress less frequently
    if image_idx % (mp.cpu_count() * 2 if 'mp' in globals() and mp.cpu_count() > 0 else 4) == 0 :
        print(f"[Worker {worker_pid}] Starting original image index {image_idx}...")

    # Data Preprocessing
    if not y_list_tensors: return image_idx, None, []
    rater_masks_list_np = []
    for r_idx, rater_mask_tensor in enumerate(y_list_tensors):
        if not torch.is_tensor(rater_mask_tensor) or rater_mask_tensor.ndim != 3 or rater_mask_tensor.shape[0] != 1:
            continue
        rater_masks_list_np.append(rater_mask_tensor.squeeze(0).cpu().numpy().astype(int))
    if not rater_masks_list_np: return image_idx, None, []

    current_img_dims = rater_masks_list_np[0].shape
    current_n_raters = len(rater_masks_list_np)

    # L_initial_guess for run_mcmc_for_image will be None to ensure random start
    L_initial_for_mcmc = None # As per user's request for null initial guess

    try:
        L_samples, se_samples_img, sp_samples_img = run_mcmc_for_image(
            rater_masks_list_np, current_img_dims, current_n_raters,
            L_initial_guess=L_initial_for_mcmc,
            n_global_steps=n_global_mcmc_steps,
            n_gibbs_sweeps_local=n_gibbs_sweeps_local,
            n_burn_in_global=n_global_burn_in,
            image_idx_for_print=image_idx
        )

        L_posterior_mean = None
        rater_params_current_img = []
        if L_samples:
            L_posterior_mean = np.mean(L_samples, axis=0)
            estimated_se_img = [np.mean(s) for s in se_samples_img]
            estimated_sp_img = [np.mean(s) for s in sp_samples_img]
            rater_params_current_img = [{'Se': se_k, 'Sp': sp_k} for se_k, sp_k in zip(estimated_se_img, estimated_sp_img)]
        return image_idx, L_posterior_mean, rater_params_current_img
    except Exception as e_worker:
        print(f"[Worker {worker_pid}, Img Idx {image_idx}] ERROR during MCMC: {e_worker}")
        import traceback
        traceback.print_exc()
        return image_idx, None, []


# --- Main Execution with Multiprocessing ---
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Note: Multiprocessing start method already set or could not be set to 'spawn'.")

    # --- USER: Instantiate Your PyTorch Dataset ---
    # ... (Your dataset instantiation logic - same as before) ...
    my_ar_dataset_instance = None # Placeholder
    try:
        # Replace with your actual dataset path and var_names
        # from pathlib import Path # Ensure Path is imported if used in ARMultiAnnDataset
        ROOT_PATH_ACTUAL = "/home/sbk29/data/AR/mini_test"
        VAR_NAMES_ACTUAL = ["TMQ"]
        my_ar_dataset_instance = ARMultiAnnDataset(root=ROOT_PATH_ACTUAL, var_names=VAR_NAMES_ACTUAL)
        if len(my_ar_dataset_instance) == 0:
            print("User dataset initialized but found 0 samples. Check dataset path and logic.")
            my_ar_dataset_instance = None
        pass # This line should be replaced by your dataset instantiation
    except Exception as e:
        print(f"Could not instantiate user's ARMultiAnnDataset: {e}. ")
        my_ar_dataset_instance = None

    
    # (Using placeholder data for brevity, replace with your actual dataset instantiation)
    if my_ar_dataset_instance is None:
        print("Using placeholder data...")
        num_dataset_images = 10; IMG_DIMS_DATASET = (20, 30); N_RATERS_DATASET_MIN, N_RATERS_DATASET_MAX = 2, 4
        placeholder_dataset_main = []
        for i in range(num_dataset_images):
            x_vars_ph = torch.randn(2, IMG_DIMS_DATASET[0], IMG_DIMS_DATASET[1])
            num_r = np.random.randint(N_RATERS_DATASET_MIN, N_RATERS_DATASET_MAX + 1)
            y_list_ph = [(torch.rand(1, IMG_DIMS_DATASET[0], IMG_DIMS_DATASET[1]) > 0.7).float() for _ in range(num_r)]
            y_cons_ph = torch.stack(y_list_ph).mean(0) if y_list_ph else torch.zeros(1, IMG_DIMS_DATASET[0], IMG_DIMS_DATASET[1])
            placeholder_dataset_main.append((x_vars_ph, y_list_ph, y_cons_ph))
        data_source_main = placeholder_dataset_main; data_source_len_main = len(placeholder_dataset_main)
        data_source_getitem_main = lambda idx_ph: placeholder_dataset_main[idx_ph]
    else:
        data_source_main = my_ar_dataset_instance; data_source_len_main = len(my_ar_dataset_instance)
        data_source_getitem_main = lambda idx_ds: my_ar_dataset_instance[idx_ds]
    # --- End Dataset Handling ---

    # New MCMC Hyperparameters
    N_GLOBAL_MCMC_STEPS = 100  # Total global Metropolis-Hastings steps
    N_GIBBS_SWEEPS_LOCAL = 5   # Number of local Gibbs sweeps to generate a candidate L
    N_GLOBAL_BURN_IN = 10      # Burn-in period for the global MCMC chain

    NUM_CORES = min(40, mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1) # Reduce for testing this complex change
    print(f"Number of CPU cores to be used by Pool: {NUM_CORES}")

    tasks_args_list = []
    for i in range(data_source_len_main):
        try:
            x_vars_tensor, y_list_tensors, y_cons_tensor = data_source_getitem_main(i)
            # Pass new MCMC parameters to the worker
            tasks_args_list.append((i, x_vars_tensor, y_list_tensors, y_cons_tensor,
                                    N_GLOBAL_MCMC_STEPS, N_GIBBS_SWEEPS_LOCAL, N_GLOBAL_BURN_IN))
        except Exception as e_fetch:
            print(f"Error fetching data for image index {i}: {e_fetch}. Skipping this image.")

    print(f"\nStarting parallel processing of {len(tasks_args_list)} images using {NUM_CORES} cores...")
    print(f"MCMC settings: Global Steps={N_GLOBAL_MCMC_STEPS}, Local Gibbs Sweeps={N_GIBBS_SWEEPS_LOCAL}, Global Burn-in={N_GLOBAL_BURN_IN}")
    print(f"LAMBDA_SMOOTH={LAMBDA_SMOOTH}, LAMBDA_SHAPE={LAMBDA_SHAPE}")


    results_from_pool = []
    with mp.Pool(processes=NUM_CORES) as pool:
        results_from_pool = pool.map(process_image_worker, tasks_args_list)

    # --- Process and Store Results (same as before) ---
    all_estimated_L_masks_posterior_mean = [None] * data_source_len_main
    all_estimated_rater_params_per_image = [None] * data_source_len_main
    successful_processes = 0
    for result_item in results_from_pool:
        if result_item is not None:
            original_idx, L_mean, rater_params = result_item
            if L_mean is not None:
                all_estimated_L_masks_posterior_mean[original_idx] = L_mean
                all_estimated_rater_params_per_image[original_idx] = rater_params
                successful_processes +=1
    print(f"\n--- Overall Parallel Processing Complete ---")
    print(f"Successfully processed results for {successful_processes}/{len(tasks_args_list)} images.")

    # --- Visualization (same as before, ensure you use data_source_getitem_main for original data) ---
    # ... (your visualization code) ...
    # (Example for the first successfully processed image)
    first_successful_plot_idx = -1
    for idx_plot, l_mask_plot in enumerate(all_estimated_L_masks_posterior_mean):
        if l_mask_plot is not None:
            first_successful_plot_idx = idx_plot
            break
    if first_successful_plot_idx != -1:
        print(f"\nVisualizing results for original Image Index {first_successful_plot_idx}")
        _, y_list_tensors_plot, y_cons_tensor_plot = data_source_getitem_main(first_successful_plot_idx) # Get original data
        rater_masks_plot_np_viz = [m.squeeze(0).cpu().numpy() for m in y_list_tensors_plot]
        num_plot_cols_viz = len(rater_masks_plot_np_viz) + 2
        fig, axes_viz = plt.subplots(1, num_plot_cols_viz, figsize=(min(20, 4 * num_plot_cols_viz), 4)) # Cap width
        if num_plot_cols_viz == 1: axes_viz = np.array([axes_viz])
        for k_plot_viz in range(len(rater_masks_plot_np_viz)):
            axes_viz[k_plot_viz].imshow(rater_masks_plot_np_viz[k_plot_viz], cmap='gray', interpolation='nearest')
            axes_viz[k_plot_viz].set_title(f"Img Idx {first_successful_plot_idx} - Rater {k_plot_viz+1}")
            axes_viz[k_plot_viz].axis('off')
        axes_viz[len(rater_masks_plot_np_viz)].imshow(y_cons_tensor_plot.squeeze(0).cpu().numpy(), cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        axes_viz[len(rater_masks_plot__np_viz)].set_title(f"Img Idx {first_successful_plot_idx} - Original Consensus")
        axes_viz[len(rater_masks_plot_np_viz)].axis('off')
        axes_viz[len(rater_masks_plot_np_viz)+1].imshow(all_estimated_L_masks_posterior_mean[first_successful_plot_idx], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        axes_viz[len(rater_masks_plot_np_viz)+1].set_title(f"Img Idx {first_successful_plot_idx} - Posterior Mean L")
        axes_viz[len(rater_masks_plot_np_viz)+1].axis('off')
        plt.tight_layout()
        plt.show()
