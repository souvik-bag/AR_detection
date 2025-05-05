import os
import pathlib
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
import psutil
from climatenet.utils.data import ClimateDataset
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events
import torch
import numpy as np
from os import path
from climatenet.procrustes_loss_new import ProcrustesLossBag
config = Config("/home/sbk29/data/github_AR/AR_detection/climatenet/config_new.json")
# print(config.num_classes)  # should be 3
cgnet = CGNet(config)

train_path = '/home/sbk29/data/AR/'
train_dataset = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
cgnet.train_procrustes(train_dataset)

cgnet.save_model('/home/sbk29/data/ClimateNet/climatenet/weights/')