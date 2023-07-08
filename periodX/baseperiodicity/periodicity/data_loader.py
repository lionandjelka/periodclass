import pandas as pd
from multiprocessing import Manager

class DataLoader:
    def __init__(self, path_source, path_obj, shared_data):
        self.path_source = path_source
        self.path_obj = path_obj
        self.shared_data = shared_data
        self.fs_gp = None
        self.fs_df = None
        self.object_df = None
        self.td_objects = None
    
    def load_fs_df(self):
        self.shared_data['fs_df'] = pd.read_parquet(self.path_source)
    
    def load_fs_gp(self):
        fs_df = self.shared_data['fs_df']
        self.fs_gp = fs_df.groupby('objectId')
        self.shared_data['fs_gp'] = self.fs_gp
    
    def load_object_df(self):
        self.shared_data['object_df'] = pd.read_parquet(self.path_obj)
    
    def get_loaded_data(self):
        fs_df = self.shared_data['fs_df']
        object_df = self.shared_data['object_df']
        lc_cols = [col for col in object_df.columns if 'Periodic' in col]
        self.td_objects = object_df.dropna(subset=lc_cols, how='all').copy()
        
        self.fs_gp = self.shared_data['fs_gp']
        
        return fs_df, object_df, self.td_objects, self.fs_gp

def load_data(path_source, path_obj, shared_data):
    manager = Manager()
    shared_data = manager.dict(shared_data)
    
    loader = DataLoader(path_source, path_obj, shared_data)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    fs_df, object_df, td_objects, fs_gp = loader.get_loaded_data()
    
    shared_data['fs_df'] = fs_df
    shared_data['object_df'] = object_df
    shared_data['td_objects'] = td_objects
    shared_data['fs_gp'] = fs_gp
    
    return shared_data['fs_df'], shared_data['object_df'], shared_data['td_objects'], shared_data['fs_gp']






