# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import sys

from torch.utils.data import Dataset, DataLoader


class CorrectFileDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        lines = correct_files(path)
        create_new_file(f'{path}.new', lines)
        return path


def correct_files(path):
    with open(path) as file:
        lines = file.readlines()
        i = 0
        correct_lines = []
        while i < len(lines):
            line = lines[i].strip()
            if 'id' in line:
                row = [line]
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    row.append(line)
                    if ']' in line:
                        row = ' '.join(row)
                        row = row.replace('[', '"[')
                        row = row.replace(']', ']"')
                        correct_lines.append(row)
                        break
                    i += 1

            i += 1
        return correct_lines


def create_new_file(path, lines):
    with open(path, mode="a+") as file:
        for line in lines:
            file.write(f"{line}\n")

    print(f"CREATED: {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    paths = ['data/vec_vox1.txt', 'data/vec_vox2.txt']
    dataset = CorrectFileDataset(paths)
    loader = DataLoader(dataset, num_workers=2, batch_size=1)
    for i, x in enumerate(loader, start=1):
        print(f"Batch [{i}/{len(loader)}]")
        sys.stdout.flush()
