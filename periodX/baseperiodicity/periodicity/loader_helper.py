from multiprocessing import Manager
from periodicity.data_loader import DataLoader
from periodicity import globalss

def load_and_get_data(path_source, path_obj, shared_data):
    manager = Manager()
    shared_data = manager.dict()
    
    loader = DataLoader(path_source, path_objects, shared_data)
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()
    
    # Access the loaded DataFrames and fs_gp using the get_loaded_data method
    fs_df, object_df, td_objects, fs_gp = loader.get_loaded_data()
    return fs_df, object_df, td_objects, fs_gp
