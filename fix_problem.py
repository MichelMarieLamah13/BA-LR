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
        df = pd.read_csv(self.files[i])
        if todelete in df.columns:
            df = df.drop([todelete], axis=1)
            df.to_csv(path, index=False)
        return df.columns.tolist()


if __name__ == "__main__":
    path = 'data/BA/*.csv'
    dataset = DropDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4)
    for i, x in enumerate(dataloader):
        print(f"Batch {i + 1}")
        sys.stdout.flush()
        print(f"{x}, {len(x)}")
        sys.stdout.flush()
