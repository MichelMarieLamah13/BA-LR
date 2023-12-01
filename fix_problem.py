# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import glob
import os.path
import pdb
import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader

import multiprocessing


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
    print("START correct vox1 opensmile")
    sys.stdout.flush()
    path = 'data/vox1_opensmile.csv'
    df = pd.read_csv(path)
    todelete = "Unnamed: 0"
    if todelete in df.columns:
        df = df.drop([todelete], axis=1)
    df['name'] = df['name'].apply(create_name)
    df.to_csv(f'{path}.new')
    print("END")
    sys.stdout.flush()


def correct_dout_dtyp():
    print("START correct dout dtyp")
    sys.stdout.flush()
    paths = ['data/dout_clean.txt', 'data/typ_clean.txt']
    for path in paths:
        with open(path) as file:
            lines = file.readlines()
            keys = []
            values = []
            for line in lines:
                parts = line.split(':')
                ba = parts[0].strip()
                value = parts[1].strip()

                keys.append(ba)
                values.append(value)

        df = pd.DataFrame({'ba': keys, 'value': values})
        df.to_csv(f'{path}.new', index=False)

    print("END")
    sys.stdout.flush()


def create_df_binary():
    pdb.set_trace()
    path = "data/vec_vox1.txt.new"
    save_path = "data/df_binary.csv"
    pass


if __name__ == "__main__":
    # create_df_binary()

    # Create two separate processes for each function
    process1 = multiprocessing.Process(target=correct_vox1_opensmile)
    process2 = multiprocessing.Process(target=correct_dout_dtyp)

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()
