from gomoku import Position

out_file = "train_data.txt"
files = ["train_data_1.txt", "train_data_2.txt"]

with open (out_file, "w") as f:
    for path in files:
        with open(path, "r") as infile:
            for line in infile:
                f.write(line)

with open (out_file, "r") as f:
    print(len(f.readlines()))


    