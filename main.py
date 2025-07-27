import time
import random
from gomoku import Position, IntersectionState
from gtp import GTP
from uct_engine import UCTEngine

gtp = GTP(UCTEngine())
gtp.mainloop("uct_log.txt")