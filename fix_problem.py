# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import glob
import os.path
import pdb
import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DropDataset(Dataset):
    def __init__(self, path):
        self.files = glob.glob(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        todelete = "Unnamed: 0"
        path = self.files[i]
        df = pd.read_csv(self.files[i])
        if todelete in df.columns:
            df = df.drop([todelete], axis=1)
            df.to_csv(path, index=False)
        return df.columns.tolist()


def delete_columns():
    path = 'data/BA/*.csv'
    dataset = DropDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4)
    for i, x in enumerate(dataloader):
        print(f"Batch {i + 1}")
        sys.stdout.flush()
        print(f"{x}, {len(x)}")
        sys.stdout.flush()


def remove_space():
    path = "data/vox2_meta.csv"
    df = pd.read_csv(path)
    columns = [c.strip() for c in df.columns.tolist()]
    df.columns = columns
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.to_csv(path, index=False)


def create_name(path: str):
    parts = path.split('.')[0]
    parts = parts.split('/')[-3:]
    fname = '-'.join(parts)
    return fname


def correct_vox1_opensmile():
    path = 'data/vox1_opensmile.csv'
    df = pd.read_csv(path)
    todelete = "Unnamed: 0"
    if todelete in df.columns:
        df = df.drop([todelete], axis=1)

    df.rename(columns={'name': 'path'}, inplace=True)
    df['name'] = df['path'].apply(create_name)

    df.to_csv(path)


def create_df_binary():
    pdb.set_trace()
    path = "data/vec_vox1.txt.new"
    save_path = "data/df_binary.csv"
    pass


if __name__ == "__main__":
    create_df_binary()
