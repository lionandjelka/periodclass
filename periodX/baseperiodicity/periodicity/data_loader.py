import pandas as pd
from multiprocessing import Manager
from periodicity import globalss

class DataLoader:
    def __init__(self, path_source, path_obj, shared_data):
        self.path_source = path_source
        self.path_obj = path_obj
        self.shared_data = shared_data
    
    def load_fs_df(self):
        # Set the file path for fs_df in the shared_data
        self.shared_data['fs_df'] = pd.read_parquet(self.path_source)
    
    def load_fs_gp(self):
        # Load fs_df from the file path and create fs_gp
        fs_df = self.shared_data['fs_df']
        self.shared_data['fs_gp'] = fs_df.groupby('objectId')
    
    def load_object_df(self):
        # Set the file path for object_df and td_objects in the shared_data
        self.shared_data['object_df'] = pd.read_parquet(self.path_obj)
    
    def get_loaded_data(self):
        # Load the actual data (fs_df, object_df, td_objects, fs_gp) when needed
        fs_df = self.shared_data['fs_df']
        object_df = self.shared_data['object_df']
        lc_cols = [col for col in object_df.columns if 'Periodic' in col]
        td_objects = object_df.dropna(subset=lc_cols, how='all').copy()
        fs_gp = self.shared_data['fs_gp']
        
        return fs_df, object_df, td_objects, fs_gp

def load_data(path_source, path_obj):
    manager = Manager()
    shared_data = manager.dict()
    
    loader = DataLoader(path_source, path_obj, shared_data)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    fs_df, object_df, td_objects, fs_gp = loader.get_loaded_data()
    
    # Assign the loaded data to the global variables in the globalss module
    globalss.fs_df = fs_df
    globalss.object_df = object_df
    globalss.td_objects = td_objects
    globalss.fs_gp = fs_gp
    
    return globalss.fs_df, globalss.object_df, globalss.td_objects, globalss.fs_gp




