# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb


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
        pdb.set_trace()
        return correct_lines


def create_new_file(path, lines):
    with open(path, mode="a+") as file:
        for line in lines:
            file.write(f"{line}\n")

    print(f"CREATED: {path} ")


if __name__ == "__main__":
    paths = ['tp_jef/vec_vox1.txt', 'tp_jef/vec_vox2.txt']
    for path in paths:
        lines = correct_files(path)
        create_new_file(f'{path}.new', lines)

