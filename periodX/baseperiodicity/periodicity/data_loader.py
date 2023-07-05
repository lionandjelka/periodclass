import pandas as pd
from multiprocessing import Manager

class DataLoader:
    def __init__(self, path_source, path_obj, shared_data):
        self.path_source = path_source
        self.path_obj = path_obj
        self.shared_data = shared_data
    
    def load_fs_df(self):
        # Set the file path for fs_df in the shared_data
        self.shared_data['fs_df_path'] = self.path_source
    
    def load_fs_gp(self):
        # Load fs_df from the file path and store the necessary information for fs_gp
        fs_df_path = self.shared_data['fs_df_path']
        fs_df = pd.read_parquet(fs_df_path)
        self.shared_data['fs_df_groupby_column'] = 'objectId'
    
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
        
        fs_gp = fs_df.groupby(self.shared_data['fs_df_groupby_column'])
        
        return fs_df, object_df, td_objects, fs_gp
    


#if __name__ == '__main__':
#    manager = Manager()
#    shared_data = manager.dict()
    
#    loader = DataLoader('path/to/source', 'path/to/objects', shared_data)
#    loader.load_fs_df()
#    loader.load_fs_gp()
#    loader.load_object_df()
    
    # Access the loaded DataFrames and fs_gp using the get_loaded_data method
#    fs_df, object_df, td_objects, fs_gp = loader.get_loaded_data()
    
    # Use fs_df, object_df, td_objects, fs_gp in other classes/functions as needed

