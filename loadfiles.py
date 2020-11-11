from pathlib import Path


def loadfile():
    try:
        file = open("filenames.txt", "r")
    except IOError:
        print("Could not open file!\n")
        return -1

    data = list()
    for line in file:
        path = Path(line.rstrip("\n"))
        if path.is_file():
            data.append(line.rstrip("\n"))

    file.close()
    return data
