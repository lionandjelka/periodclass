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
        self.shared_data['fs_df_path'] = self.path_source
    
    def load_fs_gp(self):
        # Load fs_df from the file path and create fs_gp
        fs_df_path = self.shared_data['fs_df_path']
        fs_df = pd.read_parquet(fs_df_path)
        fs_gp = fs_df.groupby('objectId')
        self.shared_data['fs_gp'] = fs_gp
    
    def load_object_df(self):
        # Set the file path for object_df and td_objects in the shared_data
        self.shared_data['object_df_path'] = self.path_obj
    
    def get_loaded_data(self):
        # Load the actual data (fs_df, object_df, td_objects, fs_gp) when needed
        fs_df_path = self.shared_data['fs_df_path']
        fs_df = pd.read_parquet(fs_df_path)
        
        object_df_path = self.shared_data['object_df_path']
        object_df = pd.read_parquet(object_df_path)
        
        lc_cols = [col for col in object_df.columns if 'Periodic' in col]
        td_objects = object_df.dropna(subset=lc_cols, how='all').copy()
        
        fs_gp = self.shared_data['fs_gp']
        
        return fs_df, object_df, td_objects, fs_gp

def load_data(path_source, path_obj, shared_data):
    loader = DataLoader(path_source, path_obj, shared_data)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    # Assign the loaded data to the global variables in globalss module
    globalss.fs_df, globalss.object_df, globalss.td_objects, globalss.fs_gp = loader.get_loaded_data()

    return globalss.fs_df, globalss.object_df, globalss.td_objects, globalss.fs_gp
# Call load_data function to load the data
#load_data(path_source, path_obj, shared_data)

# Access the loaded fs_gp from globalss module
#print(globalss.fs_gp)





