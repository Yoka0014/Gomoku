import time
import random
from gomoku import Position, IntersectionState

root_pos = Position(9, True)

start = time.perf_counter()
MAX = 10000
pos = Position(root_pos.size, True)
for i in range(MAX):
    root_pos.copy_to(pos)
    while pos.winner == IntersectionState.EMPTY and pos.empty_count > 0:
        board = pos.get_board_as_numpy()

        empties = list(pos.enumerate_empties())
        empty = random.choice(empties)
        pos.update(empty)
end = time.perf_counter()

print(MAX / (end - start), "per second")