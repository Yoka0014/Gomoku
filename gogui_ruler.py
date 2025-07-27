"""GoGui(https://github.com/Remi-Coulom/gogui?tab=readme-ov-file)でGomokuをプレイ可能にするためのルーラープログラム
"""

from engine import Engine
from gtp import GTPError

class Ruler(Engine):
    def __init__(self):
        super().__init__("GomokuRuler", "1.0")

    def gen_move(self) -> int:
        raise GTPError("Ruler does not support move generation.")

    def reg_gen_move(self) -> int:
        raise GTPError("Ruler does not support move generation.")