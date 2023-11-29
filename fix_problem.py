# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pandas as pd


def drop_first_column(path):
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    BA = [f"BA{i}" for i in range(256)]
    for ba in BA:
        for i in range(2):
            path = f"data/BA/{ba}_{i}.csv"
            drop_first_column(path)
