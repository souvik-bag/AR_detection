import logging
import numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

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





def masks_to_list(rater_masks):
    """
    rater_masks: list of R tensors [H,W] of 0/1
    returns: nested list [task][worker] -> [label]
    """
    R, H, W = len(rater_masks), *rater_masks[0].shape
    dataset_list = []

    # flatten pixel index: task_id = r * H + c
    for r_pix in range(H):
        for c_pix in range(W):
            worker_labels = []
            for m in rater_masks:
                label = int(m[r_pix, c_pix])
                worker_labels.append([label])  # wrap in list so list2array counts it
            dataset_list.append(worker_labels)
    return dataset_list, H, W





def list2array(class_num, dataset_list):
    task_num, worker_num, class_num = len(dataset_list), len(dataset_list[0]), class_num
    dataset_tensor = np.zeros((task_num, worker_num, class_num))

    for task_i in range(task_num): 
        for worker_j in range(worker_num): 
            for predict_label_k in dataset_list[task_i][worker_j]:
                dataset_tensor[task_i][worker_j][predict_label_k] += 1

    return dataset_tensor

class DawidSkeneModel:
    def __init__(self,
                 class_num,
                 max_iter = 100,
                 tolerance = 0.01) -> None:
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance

    def run(self, dataset):
        self.task_num, self.worker_num, _ = dataset.shape
        self.dataset_tensor = dataset
        predict_label =  self.dataset_tensor.sum(1) / self.dataset_tensor.sum(1).sum(1).reshape(-1, 1)

        flag = True
        prev_error_rates, prev_predict_label = None, None
        iter_num = 0

        while flag:
            error_rates = self._m_step(predict_label)
            next_predict_label = self._e_step(predict_label, error_rates)
            log_L = self._get_likelihood(predict_label, error_rates)

            if iter_num == 0:
                logging.info("{}\t{}".format(iter_num, log_L))
            else:
                marginal_predict = np.sum(predict_label, 0) / self.task_num
                prev_marginal_predict = np.sum(prev_predict_label, 0) / self.task_num
                marginals_diff = np.sum(np.abs(marginal_predict - prev_marginal_predict))
                error_rates_diff = np.sum(np.abs(error_rates - prev_error_rates))

                if self._check_condition(marginals_diff, error_rates_diff, iter_num):
                    flag = False

            prev_error_rates = error_rates
            prev_predict_label = predict_label
            predict_label = next_predict_label
            iter_num += 1

        worker_reliability = {}
        for i in range(self.worker_num):
            ie_rates = marginal_predict * error_rates[i, :, :]
            reliability = np.sum(np.diag(ie_rates))
            worker_reliability[i] = reliability
            
        return marginal_predict, error_rates, worker_reliability, predict_label

    def _check_condition(self, marginals_diff, error_rates_diff, iter_num):
        return (marginals_diff < self.tolerance and error_rates_diff < self.tolerance) or iter_num > self.max_iter

    def _m_step(self, predict_label):
        error_rates = np.zeros((self.worker_num, self.class_num, self.class_num))

        # Equation 2.3
        for i in range(self.class_num):
            worker_error_rate = np.dot(predict_label[:, i], self.dataset_tensor.transpose(1, 0 ,2))
            sum_worker_error_rate = worker_error_rate.sum(1)
            sum_worker_error_rate = np.where(sum_worker_error_rate == 0 , -10e9, sum_worker_error_rate)
            error_rates[:, i, :] = worker_error_rate / sum_worker_error_rate.reshape(-1,1)                                                                        
        return error_rates
    
    def _e_step(self, predict_label, error_rates):
        marginal_probability = predict_label.sum(0) / self.task_num
        next_predict_label = np.zeros([self.task_num, self.class_num])

        # Equation 2.5
        for i in range(self.task_num):
            class_likelood = self._get_class_likelood(error_rates, self.dataset_tensor[i])
            next_predict_label[i] = marginal_probability * class_likelood
            sum_marginal_probability = next_predict_label[i].sum()
            sum_marginal_probability = np.where(sum_marginal_probability == 0 , -10e9, sum_marginal_probability)
            next_predict_label[i] /= sum_marginal_probability
        return next_predict_label

    def _get_likelihood(self, predict_label, error_rates):
        log_L = 0
        marginal_probability = predict_label.sum(0) / self.task_num

        # Equation 2.7
        for i in range(self.task_num):
            class_likelood = self._get_class_likelood(error_rates, self.dataset_tensor[i])
            log_L += np.log((marginal_probability * class_likelood).sum())
        return log_L

    def _get_class_likelood(self, error_rates, task_tensor):
        # \sum_{j=1}^J p_{j} \prod_{k=1}^K \prod_{l=1}^J\left(\pi_{j l}^{(k)}\right)^{n_{il}^{(k)}}
        return np.power(error_rates.transpose(0, 2, 1), np.broadcast_to(task_tensor.reshape(self.worker_num, self.class_num, 1), (self.worker_num, self.class_num, self.class_num))).transpose(1, 2, 0).prod(0).prod(1)
    
    
    # where to write results
OUTPUT_DIR = "/home/sbk29/data/github_AR/AR_detection/u-net/ds_results/test_val"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) the worker function
def process_image(idx, root, var_names):
    # load one “date”
    ds = ARMultiAnnDataset(root, var_names)
    x_vars, y_list, y_cons = ds[idx]

    # build per‐pixel list-of-lists
    rater_np = [m.squeeze(0).cpu().numpy().astype(int) for m in y_list]
    dataset_list, H, W = masks_to_list(rater_np)
    
    # build DS tensor & run EM
    tensor_3d = list2array(class_num=2, dataset_list=dataset_list)
    model     = DawidSkeneModel(class_num=2, max_iter=50, tolerance=1e-4)
    marginal, err_rates, reliability, post_per_task = model.run(tensor_3d)
    
    # reshape posterior into an H×W map
    posterior_map = post_per_task[:,1].reshape(H, W).astype(np.float32)
    
    # save everything
    np.save(os.path.join(OUTPUT_DIR, f"posterior_map_{idx}.npy"), posterior_map)
    np.save(os.path.join(OUTPUT_DIR, f"error_rates_{idx}.npy"), err_rates.astype(np.float32))
    with open(os.path.join(OUTPUT_DIR, f"reliability_{idx}.json"), "w") as fp:
        json.dump(reliability, fp)
    
    return idx, posterior_map.mean()  # return index and a quick summary

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



# 2) dispatch across cores
if __name__ == "__main__":
    ROOT = "/home/sbk29/data/AR/test_val"
    VARS = ["TMQ"]
    ds   = ARMultiAnnDataset(ROOT, VARS)
    N    = len(ds)

    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(process_image, i, ROOT, VARS) for i in range(N)]
        for fut in as_completed(futures):
            idx, avg_p = fut.result()
            print(f"✅ Image {idx} done, mean posterior={avg_p:.3f}")