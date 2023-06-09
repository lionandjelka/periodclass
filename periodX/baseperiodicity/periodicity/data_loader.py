import pandas as pd

class DataLoader:
    def __init__(self):
        self.fs_gp = None
        self.fs_df = None
        self.object_df = None
        self.td_objects = None

    def load_fs_df(self, path_source):
        # Loading will take some time ...
        self.fs_df = pd.read_parquet(path_source)
        return self.fs_df

    def load_fs_gp(self):
        # groupby forcedsource table by objectid
        self.fs_gp = self.fs_df.groupby('objectId')
        return self.fs_gp

    def load_object_df(self, path_obj):
        self.object_df = pd.read_parquet(path_obj)
        # select the objects that have time domain data
        lc_cols = [col for col in self.object_df.columns if 'Periodic' in col]
        self.td_objects = self.object_df.dropna(subset=lc_cols, how='all').copy()
        print("Data loaded and processed successfully.")
        return self.td_objects

    def get_fs_df(self):
        return self.fs_df

    def get_fs_gp(self):
        return self.fs_gp

    def get_object_df(self):
        return self.object_df

    def get_td_objects(self):
        return self.td_objects
