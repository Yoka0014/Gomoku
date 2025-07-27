import time
import random
from gomoku import Position, IntersectionState
from gtp import GTP
from gogui_ruler import Ruler

if __name__ == "__main__":
    gtp = GTP(Ruler())
    gtp.mainloop("log.txt")