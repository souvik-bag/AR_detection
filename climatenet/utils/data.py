from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 

    Parameters
    ----------
    path : str
        The path to the directory containing the dataset (in form of .nc files)
    config : Config
        The model configuration. This allows to automatically infer the fields we are interested in 
        and their normalisation statistics

    Attributes
    ----------
    path : str
        Stores the Dataset path
    fields : dict
        Stores a dictionary mapping from variable names to normalisation statistics
    files : [str]
        Stores a sorted list of all the nc files in the Dataset
    length : int
        Stores the amount of nc files in the Dataset
    '''
  
    def __init__(self, path: str, config: Config):
        self.path: str = path
        self.fields: dict = config.fields
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        for variable_name, stats in self.fields.items():   
            var = features.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset)

    @staticmethod
    def collate(batch):
        return xr.concat(batch, dim='time')

# class ClimateDatasetLabeled(ClimateDataset):
#     '''
#     The labeled Climate Dataset class. 
#     Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
#     '''

#     def __getitem__(self, idx: int):
#         file_path: str = path.join(self.path, self.files[idx]) 
#         dataset = xr.load_dataset(file_path)
#         labels = dataset['LABELS']
#         transformed_labels = xr.where(labels == 2, 1, 0)
#         return self.get_features(dataset),transformed_labels

#     @staticmethod 
#     def collate(batch):
#         data, labels = map(list, zip(*batch))
#         return xr.concat(data, dim='time'), xr.concat(labels, dim='time')
    
    
    
class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class. 
    Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
    '''

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        labels = dataset['LABELS']
        transformed_labels = xr.where(labels == 2, 1, 0)
        
        
        return self.get_features(dataset), transformed_labels

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')

# class ClimateDatasetLabeled(ClimateDataset):
#     '''
#     The labeled Climate Dataset class.
#     Corresponds to the normal Climate Dataset but returns labels as well and batches accordingly.
    
#     Original labels:
#       0: Background
#       1: TC (to be discarded/treated as background)
#       2: AR (to be considered as the object of interest)
    
#     In this class, we convert:
#       - All pixels with label 1 become 0 (background)
#       - All pixels with label 2 become 1 (object)
#     '''
    
#     def __getitem__(self, idx: int):
#         file_path: str = path.join(self.path, self.files[idx]) 
#         dataset = xr.load_dataset(file_path)
#         features = self.get_features(dataset)
#         labels = dataset['LABELS']
        
#         # Transform the labels:
#         # - Where labels equal 2 (AR) becomes 1
#         # - Otherwise (background or TC), set to 0.
#         transformed_labels = xr.where(labels == 2, 1, 0)
        
#         return features, transformed_labels
    
#      # Debug: check the types/shapes of items being returned
#     print("Features shape:", features.shape)
#     print("Labels shape:", transformed_labels.shape)
#     # If you're inadvertently returning a third thing, check if it comes from additional processing.

#     @staticmethod 
#     def collate(batch):
#         data, labels = map(list, zip(*batch))
#         return xr.concat(data, dim='time'), xr.concat(labels, dim='time')
