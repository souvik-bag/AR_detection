###########################################################################
# CGNet: A Light-weight Context Guided Network for Semantic Segmentation
# Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from climatenet.modules import *
# from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
# from climatenet.utils.losses import jaccard_loss
# from climatenet.utils.metrics import get_cm, get_iou_perClass
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import xarray as xr
# from climatenet.utils.utils import Config
# from os import path
# import pathlib

# class CGNet():
#     '''
#     The high-level CGNet class.
#     This allows training and running CGNet without interacting with PyTorch code.
#     If you are looking for a higher degree of control over the training and inference,
#     we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

#     Parameters
#     ----------
#     config : Config
#         The model configuration.
#     model_path : str
#         Path to load the model and config from.

#     Attributes
#     ----------
#     config : dict
#         Stores the model config
#     network : CGNetModule
#         Stores the actual model (nn.Module)
#     optimizer : torch.optim.Optimizer
#         Stores the optimizer we use for training the model
#     '''

#     def __init__(self, config: Config = None, model_path: str = None):

#         if config is not None and model_path is not None:
#             raise ValueError('''Config and weight path set at the same time.
#             Pass a config if you want to create a new model,
#             and a weight_path if you want to load an existing model.''')

#         if config is not None:
#             # Create new model
#             self.config = config
#             self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
#         elif model_path is not None:
#             # Load model
#             self.config = Config(path.join(model_path, 'config.json'))
#             self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
#             self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
#         else:
#             raise ValueError('''You need to specify either a config or a model path.''')

#         self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)

#     # def train(self, dataset: ClimateDatasetLabeled):
#     #     '''Train the network on the given dataset for the given amount of epochs'''
#     #     self.network.train()
#     #     collate = ClimateDatasetLabeled.collate
#     #     loader = DataLoader(dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=4, shuffle=True)
#     #     for epoch in range(1, self.config.epochs+1):

#     #         print(f'Epoch {epoch}:')
#     #         epoch_loader = tqdm(loader)
#     #         aggregate_cm = np.zeros((3,3))

#     #         for features, labels in epoch_loader:

#     #             # Push data on GPU and pass forward
#     #             features = torch.tensor(features.values).cuda()
#     #             labels = torch.tensor(labels.values).cuda()

#     #             outputs = torch.softmax(self.network(features), 1)

#     #             # Update training CM
#     #             predictions = torch.max(outputs, 1)[1]
#     #             aggregate_cm += get_cm(predictions, labels, 3)

#     #             # Pass backward
#     #             loss = jaccard_loss(outputs, labels)
#     #             epoch_loader.set_description(f'Loss: {loss.item()}')
#     #             loss.backward()
#     #             self.optimizer.step()
#     #             self.optimizer.zero_grad()

#     #         print('Epoch stats:')
#     #         print(aggregate_cm)
#     #         ious = get_iou_perClass(aggregate_cm)
#     #         print('IOUs: ', ious, ', mean: ', ious.mean())
#     def train(self, dataset: ClimateDatasetLabeled):

#         '''Train the network on the given dataset for the given number of epochs'''

#         self.network.train()
#         collate = ClimateDatasetLabeled.collate
#         loader = DataLoader(dataset, batch_size=self.config.train_batch_size,
#                         collate_fn=collate, num_workers=2, shuffle=True)

#     # Create the cross entropy loss function
#         loss_fn = nn.CrossEntropyLoss()

#         for epoch in range(1, self.config.epochs+1):
#             print(f'Epoch {epoch}:')
#             epoch_loader = tqdm(loader)
#             aggregate_cm = np.zeros((3,3))  # assuming 3 classes: BG, TC, AR

#             for features, labels in epoch_loader:
#             # Move data to GPU
#                 features = torch.tensor(features.values).cuda()
#                 labels = torch.tensor(labels.values).cuda()  # shape: (B, H, W) with integer labels

#             # Forward pass: get raw logits (no softmax)
#                 logits = self.network(features)  # shape: (B, C, H, W)

#             # Compute cross entropy loss; CrossEntropyLoss applies LogSoftmax internally.
#                 loss = loss_fn(logits, labels)

#                 # Update training confusion matrix: use argmax on logits
#                 predictions = torch.argmax(logits, dim=1)  # shape: (B, H, W)
#                 aggregate_cm += get_cm(predictions, labels, 3)

#                 epoch_loader.set_description(f'Loss: {loss.item():.4f}')
#                 loss.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#             print('Epoch stats:')
#             print(aggregate_cm)
#             ious = get_iou_perClass(aggregate_cm)
#             print('IOUs: ', ious, ', mean: ', ious.mean())

#     def predict(self, dataset: ClimateDataset, save_dir: str = None):
#         '''Make predictions for the given dataset and return them as xr.DataArray'''
#         self.network.eval()
#         collate = ClimateDataset.collate
#         loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
#         epoch_loader = tqdm(loader)

#         predictions = []
#         for batch in epoch_loader:
#             features = torch.tensor(batch.values).cuda()

#             with torch.no_grad():
#                 # outputs = (self.network(features))
#                 outputs = torch.softmax(self.network(features), 1)   # Change it later for the predicted probabilities
#             preds = torch.max(outputs, 1)[1].cpu().numpy()
#             #preds = outputs.cpu().numpy()[:,2,:,:]
#             print(preds.shape)

#             coords = batch.coords
#             del coords['variable']

#             dims = [dim for dim in batch.dims if dim != "variable"]

#             predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

#         return xr.concat(predictions, dim='time')

#     def evaluate(self, dataset: ClimateDatasetLabeled):
#         '''Evaluate on a dataset and return statistics'''
#         self.network.eval()
#         collate = ClimateDatasetLabeled.collate
#         loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=2)

#         epoch_loader = tqdm(loader)
#         aggregate_cm = np.zeros((3,3))

#         for features, labels in epoch_loader:

#             features = torch.tensor(features.values).cuda()
#             labels = torch.tensor(labels.values).cuda()

#             with torch.no_grad():
#                 outputs = torch.softmax(self.network(features), 1)
#             predictions = torch.max(outputs, 1)[1]
#             aggregate_cm += get_cm(predictions, labels, 3)

#         print('Evaluation stats:')
#         print(aggregate_cm)
#         ious = get_iou_perClass(aggregate_cm)
#         print('IOUs: ', ious, ', mean: ', ious.mean())

#     def save_model(self, save_path: str):
#         '''
#         Save model weights and config to a directory.
#         '''
#         # create save_path if it doesn't exist
#         pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

#         # save weights and config
#         self.config.save(path.join(save_path, 'config_new.json'))
#         torch.save(self.network.state_dict(), path.join(save_path, 'weights_new.pth'))

#     def load_model(self, model_path: str):
#         '''
#         Load a model. While this can easily be done using the normal constructor, this might make the code more readable -
#         we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
#         '''
#         self.config = Config(path.join(model_path, 'config_new.json'))
#         self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
#         self.network.load_state_dict(torch.load(path.join(model_path, 'weights_new.pth'),  weights_only=True))


# class CGNetModule(nn.Module):
#     """
#     CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
#     This is taken from their implementation, we do not claim credit for this.
#     """
#     def __init__(self, classes=19, channels=4, M=3, N= 21, dropout_flag = False):
#         """
#         args:
#           classes: number of classes in the dataset. Default is 19 for the cityscapes
#           M: the number of blocks in stage 2
#           N: the number of blocks in stage 3
#         """
#         super().__init__()
#         self.level1_0 = ConvBNPReLU(channels, 32, 3, 2)      # feature map size divided 2, 1/2
#         self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
#         self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

#         self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
#         self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

#         self.b1 = BNPReLU(32 + channels)

#         #stage 2
#         self.level2_0 = ContextGuidedBlock_Down(32 + channels, 64, dilation_rate=2,reduction=8)
#         self.level2 = nn.ModuleList()
#         for i in range(0, M-1):
#             self.level2.append(ContextGuidedBlock(64 , 64, dilation_rate=2, reduction=8))  #CG block
#         self.bn_prelu_2 = BNPReLU(128 + channels)

#         #stage 3
#         self.level3_0 = ContextGuidedBlock_Down(128 + channels, 128, dilation_rate=4, reduction=16)
#         self.level3 = nn.ModuleList()
#         for i in range(0, N-1):
#             self.level3.append(ContextGuidedBlock(128 , 128, dilation_rate=4, reduction=16)) # CG block
#         self.bn_prelu_3 = BNPReLU(256)

#         if dropout_flag:
#             print("have droput layer")
#             self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(256, classes, 1, 1))
#         else:
#             self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

#         #init weights
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.find('Conv2d')!= -1:
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                 elif classname.find('ConvTranspose2d')!= -1:
#                     nn.init.kaiming_normal_(m.weight)
#                     if m.bias is not None:
#                         m.bias.data.zero_()

#     def forward(self, input):
#         """
#         args:
#             input: Receives the input RGB image
#             return: segmentation map
#         """
#         # stage 1
#         output0 = self.level1_0(input)
#         output0 = self.level1_1(output0)
#         output0 = self.level1_2(output0)
#         inp1 = self.sample1(input)
#         inp2 = self.sample2(input)

#         # stage 2
#         output0_cat = self.b1(torch.cat([output0, inp1], 1))
#         output1_0 = self.level2_0(output0_cat) # down-sampled

#         for i, layer in enumerate(self.level2):
#             if i==0:
#                 output1 = layer(output1_0)
#             else:
#                 output1 = layer(output1)

#         output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

#         # stage 3
#         output2_0 = self.level3_0(output1_cat) # down-sampled
#         for i, layer in enumerate(self.level3):
#             if i==0:
#                 output2 = layer(output2_0)
#             else:
#                 output2 = layer(output2)

#         output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

#         # classifier
#         classifier = self.classifier(output2_cat)

#         # upsample segmenation map ---> the input image size
#         out = F.interpolate(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
#         return out


###########################################################################
# CGNet: A Light-weight Context Guided Network for Semantic Segmentation
# Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from climatenet.utils.utils import Config
from os import path
import pathlib
from monai.losses import HausdorffDTLoss, DiceLoss, GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, FocalLoss
from monai.networks.utils import one_hot
from climatenet.procrustes_loss import ProcrustesLoss
from climatenet.procrustes_loss_new import ProcrustesDTLoss

class CGNet:
    """
    The high-level CGNet class.
    This allows training and running CGNet without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : CGNetModule
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    device : torch.device
        The device (CPU or GPU) on which computations will be performed.
    """

    def __init__(self, config: Config = None, model_path: str = None):
        # Define device: use GPU if available, else CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None and model_path is not None:
            raise ValueError(
                """Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model."""
            )

        if config is not None:
            # Create new model
            self.config = config
            self.network = CGNetModule(
                classes=len(self.config.labels), channels=len(list(self.config.fields))
            ).to(self.device)
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, "config.json"))
            self.network = CGNetModule(
                classes=len(self.config.labels), channels=len(list(self.config.fields))
            ).to(self.device)
            self.network.load_state_dict(
                torch.load(path.join(model_path, "weights.pth"))
            )
        else:
            raise ValueError("""You need to specify either a config or a model path.""")

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)

    # def train(self, dataset: ClimateDatasetLabeled):
    #     '''Train the network on the given dataset for the given number of epochs'''
    #     self.network.train()
    #     collate = ClimateDatasetLabeled.collate
    #     loader = DataLoader(dataset, batch_size=self.config.train_batch_size,
    #                         collate_fn=collate, num_workers=2, shuffle=True)

    #     # Create the cross entropy loss function
    #     loss_fn = nn.CrossEntropyLoss()

    #     for epoch in range(1, self.config.epochs+1):
    #         print(f'Epoch {epoch}:')
    #         epoch_loader = tqdm(loader)
    #         aggregate_cm = np.zeros((3, 3))  # assuming 3 classes: BG, TC, AR

    #         for features, labels in epoch_loader:
    #             # Move data to device (GPU if available, else CPU)
    #             features = torch.tensor(features.values).to(self.device)
    #             labels = torch.tensor(labels.values).to(self.device)  # shape: (B, H, W) with integer labels

    #             # Forward pass: get raw logits (no softmax)
    #             logits = self.network(features)  # shape: (B, C, H, W)

    #             # Compute cross entropy loss; CrossEntropyLoss applies LogSoftmax internally.
    #             loss = loss_fn(logits, labels)

    #             # Update training confusion matrix: use argmax on logits
    #             predictions = torch.argmax(logits, dim=1)  # shape: (B, H, W)
    #             aggregate_cm += get_cm(predictions, labels, 3)

    #             epoch_loader.set_description(f'Loss: {loss.item():.4f}')
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()

    #         print('Epoch stats:')
    #         print(aggregate_cm)
    #         ious = get_iou_perClass(aggregate_cm)
    #         print('IOUs: ', ious, ', mean: ', ious.mean())

    def train(self, dataset):
        """
        Train the network using a combined loss:
          total_loss = HausdorffDTLoss (boundary loss) + DiceLoss (overlap loss)

        The Hausdorff loss uses:
          - softmax=True (applied internally)
          - to_onehot_y=False (we provide the one-hot labels manually)
          - include_background=False (ignores the background class)
          - reduction="mean" (averages over the batch)

        The Dice loss uses:
          - softmax=True and to_onehot_y=True, so it automatically converts integer labels.

        The ground truth labels are expected to have shape [B, H, W].
        """

        self.network.train()
        collate = dataset.collate  # assuming dataset has a collate function
        loader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=collate,
            num_workers=8,
            shuffle=True,
        )

        # Initialize the loss functions.
        # For Hausdorff, we will supply one-hot labels ourselves.
        # hausdorff_loss_fn = HausdorffDTLoss(
        #     softmax=True,
        #     to_onehot_y=False,
        #     include_background=False,
        #     reduction="mean"
        # )
        
        procrustes_loss_fn = ProcrustesDTLoss(
            softmax=True,
            to_onehot_y=False,
            include_background=False,
            reduction="mean"
        )
        # # DiceLoss will convert labels to one-hot internally.
        # dice_loss_fn = FocalLoss(
        #     softmax=True,
        #     to_onehot_y=False,
        #     include_background=False,  # set to False if you want to exclude BG for Dice as well
        #     reduction="mean",
        # )
        # dice_loss_fn = FocalLoss(
        #     use_softmax=True,
        #     to_onehot_y=False,
        #     include_background=False,  # set to False if you want to exclude BG for Dice as well
        #     reduction="mean",
        # )
        
        
        # procrustes_loss_fn = ProcrustesLoss(
        #     threshold=0.8, allow_scaling=False, penalty_constant=1000
        # )

        for epoch in range(1, self.config.epochs + 1):
            print(f"Epoch {epoch}:")
            epoch_loader = tqdm(loader)
            # assuming 3 classes: BG, TC, AR  # Actually two classes now
            aggregate_cm = np.zeros((2, 2))

            for features, labels in epoch_loader:
                # Move data to device (GPU if available, else CPU)
                features = torch.tensor(features.values).to(self.device)
                labels = torch.tensor(labels.values).to(
                    self.device
                )  # expected shape: [B, H, W]
                # labels2 = torch.tensor(labels.values).long().to(self.device)
                
               
            
                

                # If labels have no channel dimension, add one for one-hot conversion for Hausdorff loss.
                if labels.dim() == 3:
                    labels2 = labels[:, None, ...]  # now shape: [B, 1, H, W]
                    # print(labels2.shape)

                # Create one-hot version for Hausdorff loss.
                # We assume the number of classes is 3 (BG, TC, AR)
                target_onehot = one_hot(labels2, num_classes=2).to(
                    self.device
                )  # shape: [B, 3, H, W]

                # Forward pass: get raw logits with shape [B, C, H, W]
                logits = self.network(features)
                # outputs = torch.softmax(self.network(features), 1)
                # criterion = nn.CrossEntropyLoss()
                # loss = criterion(logits, labels)
                

                # # Compute losses:
                # # Hausdorff loss uses our precomputed one-hot target.
                loss_procrustes = procrustes_loss_fn(logits, target_onehot)
                # # Dice loss uses the original labels (DiceLoss will one-hot encode internally).
                # # squeeze to get [B, H, W]
                # loss = dice_loss_fn(logits, target_onehot)

                # # Combine losses.
                # total_loss = loss_hausdorff
                # loss_jaccard = jaccard_loss(logits, labels)

                # # Update training confusion matrix: use argmax over logits.
                # predictions = torch.argmax(logits, dim=1)  # shape: [B, H, W]
                # aggregate_cm += get_cm(predictions, labels.squeeze(1), 3)

                epoch_loader.set_description(f"Loss: {loss_procrustes.item():.4f}")
                loss_procrustes.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # print("Epoch stats:")
            # print(aggregate_cm)
            # ious = get_iou_perClass(aggregate_cm)
            # print("IOUs:", ious, ", mean:", ious.mean())

    def train_procrustes(self, dataset):
        """
        Train the network using a combined loss:
          total_loss = DiceLoss (overlap loss) + ProcrustesLoss (boundary loss on AR)

        Assumptions:
          - The ground truth labels have shape [B, H, W] and use:
                0 for background
                1 for AR
          - The network outputs logits with shape [B, 2, H, W].
          - DiceFocalLoss will be applied on a one-hot encoded target (2 channels).
          - ProcrustesLoss is computed only on the AR channel (channel index 1).
        """

        self.network.train()
        collate = dataset.collate  # assuming dataset has a collate function
        loader = DataLoader(
        dataset,
        batch_size=self.config.train_batch_size,
        collate_fn=collate,
        num_workers=8,
        shuffle=True,
    )

    # Initialize the Dice loss function.
        # dice_loss_fn = DiceFocalLoss(
        # softmax=True,
        # to_onehot_y=True,  # Convert the target into one-hot with 2 channels.
        # include_background=False,  # Focus on the AR class.
        # reduction="mean",
        # )
        # loss_fn = nn.CrossEntropyLoss()
        # hausdorff_loss_fn = HausdorffDTLoss(
        #     softmax=True,
        #     to_onehot_y=False,
        #     include_background=False,
        #     reduction="mean"
        # )

    # Initialize the Procrustes loss function (applied on AR channel only).
        procrustes_loss_fn = ProcrustesLoss(
        threshold=0.5, allow_scaling=False, penalty_constant=500
    )

        for epoch in range(1, self.config.epochs + 1):
            print(f"Epoch {epoch}:")
            epoch_loader = tqdm(loader)
        # We use a 2x2 confusion matrix: one row and column each for background and AR.
            aggregate_cm = np.zeros((2, 2))

            for features, labels in epoch_loader:
            # Convert xarray DataArrays to torch tensors and move to device.
                features = torch.tensor(features.values).to(self.device)
                labels = torch.tensor(labels.values).to(
                self.device
            )  # expected shape: [B, H, W]
                
                print(f'labels shape : {labels.shape}')
                
            #     labels_bin = (labels == 2).long()  # now: 1 for AR, 0 for BG (and TC)
            
            # # Add channel dimension.
            #     labels_bin = labels_bin[:, None, ...]  # shape: [B, 1, H, W]
            
            # # Create one-hot target for Dice loss with 2 classes.
            #     target_onehot = one_hot(labels_bin, num_classes=2).to(self.device)  # shape: [B, 2, H, W]

            # # If labels have no channel dimension, add one.
            #     if labels.dim() == 3:
            #         labels2 = labels[:, None, ...]  # shape becomes [B, 1, H, W]
            #     else:
            #         labels2 = labels

            # # Create one-hot target for Dice loss with 2 classes.
            #     target_onehot = one_hot(labels2, num_classes=2).to(
            #     self.device
            # )  # shape: [B, 2, H, W]

            # Forward pass: get logits with shape [B, 2, H, W].
                logits = self.network(features)
                # outputs = torch.softmax(logits, 1)
                # target_onehot = one_hot(labels.unsqueeze(1), num_classes=2).to(self.device)
                # loss_ce = loss_fn(logits, labels)

            # Compute Dice loss.
                # loss_jaccard = jaccard_loss(logits, labels)

            # For Procrustes loss, compute it only for AR:
            # Compute predicted probabilities via softmax.
                pred_probs = torch.softmax(logits, dim=1)  # shape: [B, 2, H, W]
                print(f'pred_probs shape: {pred_probs.shape}')
            # Extract AR channel (channel index 1).
                pred_proba_ar = pred_probs[:, 1:2, ...]  # shape: [B, 1, H, W]
            # Create a binary ground truth mask for AR (pixels==1 indicate AR).
                gt_ar = (labels == 1).float()  # shape: [B, H, W]
                gt_ar = gt_ar[:, None, ...]  # shape: [B, 1, H, W]

            # Compute Procrustes loss.
                loss_procrustes = procrustes_loss_fn(pred_proba_ar, gt_ar)

            # Combine losses.
                # total_loss =   loss_jaccard   +  0.01 * loss_procrustes

            # # Update training confusion matrix.
            # # (Assumes get_cm() handles 2 classes correctly.)
            #     predictions = torch.argmax(logits, dim=1)  # shape: [B, H, W]
            #     aggregate_cm += get_cm(predictions, labels.squeeze(1), 2)

            #     epoch_loader.set_description(f"Loss: {total_loss.item():.4f}")
                loss_procrustes.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # print("Epoch stats:")
            # print(aggregate_cm)
            # ious = get_iou_perClass(aggregate_cm)
            # print("IOUs:", ious, ", mean:", ious.mean())

    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        """Make predictions for the given dataset and return them as xr.DataArray"""
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(
            dataset, batch_size=self.config.pred_batch_size, collate_fn=collate
        )
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values).to(self.device)

            with torch.no_grad():
                # Get raw outputs and then apply softmax for probabilities
                # outputs = torch.softmax(self.network(features), dim=1)
                outputs = self.network(features)
                
                
                print(f"Output shape : {outputs.shape}")
            # preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds = outputs.cpu().numpy()
            print(preds.shape)

            #     coords = batch.coords
            #     del coords['variable']

            #     dims = [dim for dim in batch.dims if dim != "variable"]

            #     predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

            # return xr.concat(predictions, dim='time')
            # We want an xarray dimension for classes.
            # The original 'batch' might have dims = ['time','variable','lat','lon']
            # but we no longer need 'variable'. We'll rename it to 'classes' for clarity.
            coords = dict(batch.coords)  # make a copy
            if "variable" in coords:
                del coords["variable"]

            # We'll define dims explicitly.
            # Typically, after removing 'variable', we have dims = ['time','lat','lon']
            # So let's insert 'classes' after 'time'.
            dims = ["time", "classes"] + [
                dim for dim in batch.dims if dim not in ("time", "variable")
            ]

            # Build a DataArray with shape (time, classes, lat, lon)
            # We need to create or adapt the 'classes' coordinate if we want it labeled.
            # For a typical 2-class problem, classes=[0,1]. For multi-class, classes=range(num_classes).
            num_classes = preds.shape[1]
            coords["classes"] = np.arange(num_classes)

            # The shape of 'preds' is (B, C, H, W), matching (time, classes, lat, lon).
            # So we pass coords and dims accordingly.
            pred_da = xr.DataArray(
                preds,
                coords=[
                    coords["time"],
                    coords["classes"],
                    coords["lat"],
                    coords["lon"],
                ],
                dims=dims,
                attrs=batch.attrs,
            )

            predictions.append(pred_da)

        # Concatenate along 'time'
        return xr.concat(predictions, dim="time")

    def evaluate(self, dataset: ClimateDatasetLabeled):
        """Evaluate on a dataset and return statistics"""
        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(
            dataset,
            batch_size=self.config.pred_batch_size,
            collate_fn=collate,
            num_workers=2,
        )

        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3, 3))

        for features, labels in epoch_loader:
            features = torch.tensor(features.values).to(self.device)
            labels = torch.tensor(labels.values).to(self.device)

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), dim=1)
            predictions = torch.argmax(outputs, dim=1)
            aggregate_cm += get_cm(predictions, labels, 3)

        print("Evaluation stats:")
        print(aggregate_cm)
        ious = get_iou_perClass(aggregate_cm)
        print("IOUs: ", ious, ", mean: ", ious.mean())

    def save_model(self, save_path: str):
        """
        Save model weights and config to a directory.
        """
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

        # save weights and config
        self.config.save(path.join(save_path, "config_new.json"))
        torch.save(self.network.state_dict(), path.join(save_path, "weights_new.pth"))

    def load_model(self, model_path: str):
        """
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable -
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        """
        self.config = Config(path.join(model_path, "config_new.json"))
        self.network = CGNetModule(
            classes=len(self.config.labels), channels=len(list(self.config.fields))
        ).to(self.device)
        self.network.load_state_dict(
            torch.load(path.join(model_path, "weights_new.pth"), weights_only=True)
        )


class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """

    def __init__(self, classes=19, channels=4, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        # feature map size divided 2, 1/2
        self.level1_0 = ConvBNPReLU(channels, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        # down-sample for Input Injection, factor=2
        self.sample1 = InputInjection(1)
        # down-sample for Input Injection, factor=4
        self.sample2 = InputInjection(2)

        self.b1 = BNPReLU(32 + channels)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32 + channels, 64, dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128 + channels, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have dropout layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        # Initialize weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find("ConvTranspose2d") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # Upsample segmentation map to the input image size
        out = F.interpolate(
            classifier, input.size()[2:], mode="bilinear", align_corners=False
        )
        return out
