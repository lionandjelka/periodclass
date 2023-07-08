import pandas as pd
from periodicity import globalss

class DataLoader:
    def __init__(self, path_source, path_obj):
        self.path_source = path_source
        self.path_obj = path_obj
    
    def load_fs_df(self):
        globalss.fs_df = pd.read_parquet(self.path_source)
    
    def load_fs_gp(self):
        globalss.fs_gp = globalss.fs_df.groupby('objectId')
    
    def load_object_df(self):
        globalss.object_df = pd.read_parquet(self.path_obj)
    
    def get_loaded_data(self):
        lc_cols = [col for col in globalss.object_df.columns if 'Periodic' in col]
        globalss.td_objects = globalss.object_df.dropna(subset=lc_cols, how='all').copy()
        
        return globalss.fs_df, globalss.object_df, globalss.td_objects, globalss.fs_gp

def load_data(path_source, path_obj):
    loader = DataLoader(path_source, path_obj)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    return globalss.fs_df, globalss.object_df, globalss.td_objects, globalss.fs_gp



