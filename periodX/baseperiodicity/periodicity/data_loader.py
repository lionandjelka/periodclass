import pandas as pd
from multiprocessing import Manager

# Define global variables
global fs_gp, fs_df, object_df, td_objects
fs_gp = None
fs_df = None
object_df = None
td_objects = None

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
        # Set the file path for fs_df in the shared_data
        self.shared_data['fs_df_path'] = self.path_source
    
    def load_fs_gp(self):
        # Load fs_df from the file path and store the necessary information for fs_gp
        fs_df_path = self.shared_data['fs_df_path']
        self.fs_df = pd.read_parquet(fs_df_path)
        self.shared_data['fs_df_groupby_column'] = 'objectId'
        self.fs_gp = self.fs_df.groupby(self.shared_data['fs_df_groupby_column'])
    
    def load_object_df(self):
        # Set the file path for object_df and td_objects in the shared_data
        self.shared_data['object_df_path'] = self.path_obj
    
    def get_loaded_data(self):
        # Load the actual data (fs_df, object_df, td_objects, fs_gp) when needed
        fs_df_path = self.shared_data['fs_df_path']
        self.fs_df = pd.read_parquet(fs_df_path)
        
        object_df_path = self.shared_data['object_df_path']
        self.object_df = pd.read_parquet(object_df_path)
        
        lc_cols = [col for col in self.object_df.columns if 'Periodic' in col]
        self.td_objects = self.object_df.dropna(subset=lc_cols, how='all').copy()
        
        self.fs_gp = self.fs_df.groupby(self.shared_data['fs_df_groupby_column'])

def load_data(path_source, path_obj, shared_data):
    global fs_gp, fs_df, object_df, td_objects
    
    loader = DataLoader(path_source, path_obj, shared_data)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    # Assign the loaded data to the global variables
    fs_gp = loader.fs_gp
    fs_df = loader.fs_df
    object_df = loader.object_df
    td_objects = loader.td_objects

    return fs_df, object_df, td_objects, fs_gp




