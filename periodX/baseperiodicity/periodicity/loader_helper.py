from multiprocessing import Manager
from periodicity.data_loader import DataLoader

def load_and_get_data(path_source, path_obj):
    global fs_gp, fs_df, object_df, td_objects

    # Create a shared dictionary using Manager
    manager = Manager()
    shared_data = manager.dict()

    # Initialize the DataLoader
    loader = DataLoader(path_source, path_obj, shared_data)

    # Load the data
    loader.load_fs_df()
    loader.load_fs_gp()
    loader.load_object_df()

    # Access the loaded DataFrames and fs_gp using the get_loaded_data method
    fs_df, object_df, td_objects, fs_gp = loader.get_loaded_data()

    # Assign the loaded data to the global variables
    globals()['fs_df'] = fs_df
    globals()['object_df'] = object_df
    globals()['td_objects'] = td_objects
    globals()['fs_gp'] = fs_gp
