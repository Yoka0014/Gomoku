from gomoku import Position

win_count = 0
draw_count = 0
loss_count = 0
duplicated = set()
read = set()
empty = '-' * 81
count = 0
with open("data\\train_data_qnet.txt", "r") as f:
    for line in f:
        pos, move, outcome = line.split()

        if pos == empty and move == "40":
            count += 1

print(count)