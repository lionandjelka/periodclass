from .data_loader import DataLoader

def initialize_data_loader(path_source, path_obj):
    global shared_data_loader
    shared_data_loader = DataLoader()   
    shared_data_loader.load_fs_df(path_source)
    shared_data_loader.load_fs_gp()
    shared_data_loader.load_object_df(path_obj)
