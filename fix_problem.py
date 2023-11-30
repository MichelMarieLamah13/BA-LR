# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import glob
import os.path
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
    meta_vox2 = pd.read_csv(path)
    columns = [c.strip() for c in meta_vox2.columns.tolist()]
    meta_vox2.columns = columns
    meta_vox2['Set'] = meta_vox2['Set'].apply(lambda x: x.strip())
    meta_vox2.to_csv(path, index=False)


if __name__ == "__main__":
    # enlever colonne "Unnamed: 0"
    # delete_columns()

    # meta_vox2.columns: enlever espace, colonne set enlever espace sur valeur
    remove_space()
