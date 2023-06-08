import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self):
        self.fs_gp = None
        self.fs_df = None
        self.object_df = None
        self.td_objects = None

    def load_fs_df(self, path_source):
        self.fs_df = pd.read_parquet(path_source)

    def load_fs_gp(self):
        self.fs_gp = self.fs_df.groupby('objectId')

    def load_object_df(self, path_obj):
        self.object_df = pd.read_parquet(path_obj)
        lc_cols = [col for col in self.object_df.columns if 'Periodic' in col]
        self.td_objects = self.object_df.dropna(subset=lc_cols, how='all').copy()
        print("Data loaded and processed successfully.")
