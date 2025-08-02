from enum import IntEnum
import array

import numpy as np
import torch


PASS_COORD = -2  # パスを意味する座標


class IntersectionState(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    OUT = 3


def to_opponent_color(color: IntersectionState) -> IntersectionState:
    assert color in (IntersectionState.BLACK, IntersectionState.WHITE), "Invalid color"
    return color ^ IntersectionState.WHITE


class Position:
    STONE_COUNT_TO_WIN = 5
    BOARD_SIZE_MIN = STONE_COUNT_TO_WIN
    BOARD_SIZE_MAX = 19
    MARGIN = 1  # 盤外判定を容易にするための余白
    SQRT_TABLE = [-1] * (BOARD_SIZE_MAX ** 2 + 1)   # SQRT_TABLE[i ** 2] = i を満たすテーブル.
    __CREATED_SIZE = [False] * (BOARD_SIZE_MAX + 1)  
    __MANHATTAN_TABLE_CACHE = {}
    __PADDED_COORD_CACHE = {}

    @staticmethod
    def static_init():
        for i in range(1, Position.BOARD_SIZE_MAX + 1):
            Position.SQRT_TABLE[i * i] = i

    def __init__(self, size: int, nn_input: bool=False):
        self.__size = size
        self.__padded_size = size + Position.MARGIN * 2
        self.__board = array.array('i', [IntersectionState.EMPTY] * (self.__padded_size ** 2)) 
        self.__board_numpy = np.zeros((2, self.__size, self.__size), dtype=np.float32) if nn_input else None
        self.__side_to_move = IntersectionState.BLACK
        self.__winner = IntersectionState.EMPTY
        self.__empties = (1 << (size ** 2)) - 1  # 空き交点のbitboardによる表現
        self.__empty_count = size ** 2

        # 余白部分をIntersectionState.OUTに設定
        for i in range(self.__padded_size):
            self.__board[i] = IntersectionState.OUT
            self.__board[-(self.__padded_size - i)] = IntersectionState.OUT

        for i in range(self.__size):
            y = i + Position.MARGIN
            for j in range(Position.MARGIN):
                self.__board[y * self.__padded_size + j] = IntersectionState.OUT
                self.__board[(y + 1) * self.__padded_size - j - 1] = IntersectionState.OUT

        if Position.__CREATED_SIZE[size]:
            self.__TO_PADDED_COORD = Position.__PADDED_COORD_CACHE[size]
            self.__MANHATTAN_TABLE = Position.__MANHATTAN_TABLE_CACHE[size]
            return
        
        Position.__CREATED_SIZE[size] = True

        # 通常の座標を余白も含めた座標に変換するテーブル．
        self.__TO_PADDED_COORD = [0] * (self.__size ** 2)
        for coord in range(self.__size ** 2):
            x = coord % self.__size + Position.MARGIN
            y = coord // self.__size + Position.MARGIN
            self.__TO_PADDED_COORD[coord] = x + y * self.__padded_size

        Position.__PADDED_COORD_CACHE[size] = self.__TO_PADDED_COORD

        # マンハッタン距離のテーブル
        # __MANHATTAN_TABLE[i][j] = iとjのマンハッタン距離を表す
        self.__MANHATTAN_TABLE = []
        for i in range(self.__size ** 2):
            x0, y0 = i % self.__size, i // self.__size
            dists = array.array('i', [0] * (self.__size ** 2))
            for j in range(self.__size ** 2):
                x1, y1 = j % self.__size, j // self.__size
                dists[j] = abs(x0 - x1) + abs(y0 - y1)
            self.__MANHATTAN_TABLE.append(dists)

        Position.__MANHATTAN_TABLE_CACHE[size] = self.__MANHATTAN_TABLE

    @property
    def size(self) -> int:
        return self.__size
    
    @property
    def side_to_move(self) -> IntersectionState:
        """現在の手番"""
        return self.__side_to_move
    
    @property
    def opponent_color(self) -> IntersectionState:
        """手番ではないプレイヤの石の色"""
        return self.__side_to_move ^ IntersectionState.WHITE
    
    @property
    def empty_count(self) -> int:
        """空き交点の数"""
        return self.__empty_count
    
    @property
    def winner(self) -> IntersectionState:
        """勝者. 終局していない場合はIntersectionState.EMPTY"""
        return self.__winner
    
    @property
    def is_gameover(self) -> bool:
        return self.__winner != IntersectionState.EMPTY or self.__empty_count == 0
    
    def to_tensor(self) -> torch.Tensor:
        ret = torch.from_numpy(self.__board_numpy)
        if self.__side_to_move == IntersectionState.WHITE:
            ret = ret.flip(0)
        return ret.clone()

    def copy_to(self, dest):
        """コピー先に現在の盤面をdeep copyする

        Args:
            dest (Position): コピー先

        Raises:
            TypeError: コピー先がPositionのインスタンスでない場合
            ValueError: コピー先のサイズが一致しない場合
        """
        if not isinstance(dest, Position):
            raise TypeError("Destination must be an instance of Position")
        
        if self.__size != dest.size:
            raise ValueError("Destination size does not match source size")

        dest.__board[:] = self.__board[:]
        dest.__side_to_move = self.__side_to_move
        dest.__empties = self.__empties
        dest.__empty_count = self.__empty_count
        dest.__winner = self.__winner
        if dest.__board_numpy is not None:
            if self.__board_numpy is not None:
                dest.__board_numpy = np.copy(self.__board_numpy)
            else:
                for coord in range(self.__size ** 2):
                    state = self.get_intersection_state_at(coord)
                    if state == IntersectionState.BLACK:
                        dest.__board_numpy[0, coord // self.__size, coord % self.__size] = 1.0
                    elif state == IntersectionState.WHITE:
                        dest.__board_numpy[1, coord // self.__size, coord % self.__size] = 1.0
                    else:
                        dest.__board_numpy[:, coord // self.__size, coord % self.__size] = 0.0

    def copy(self, copy_numpy=True) -> 'Position':
        new_pos = Position(self.__size, nn_input=(copy_numpy and self.__board_numpy is not None))
        self.copy_to(new_pos)
        return new_pos

    def get_intersection_state_at(self, coord: int) -> IntersectionState:
        """指定した座標の交点の状態を取得する"""
        coord = self.__TO_PADDED_COORD[coord]
        return self.__board[coord]
    
    def set_intersection_state_at(self, coord: int, state: IntersectionState):
        """指定した座標の交点の状態を設定する"""
        assert(state != IntersectionState.OUT)

        pad_coord = self.__TO_PADDED_COORD[coord]
        self.__board[pad_coord] = state
        
        if state != IntersectionState.EMPTY:
            self.__empties ^= 1 << coord
            self.__empty_count -= 1
        else:
            self.__empties |= 1 << coord
            self.__empty_count += 1

        if self.__board_numpy is not None:
            self.__board_numpy.fill(0.0)
            if state == IntersectionState.BLACK:
                self.__board_numpy[0, coord // self.__size, coord % self.__size] = 1.0
            elif state == IntersectionState.WHITE:
                self.__board_numpy[1, coord // self.__size, coord % self.__size] = 1.0
            elif state == IntersectionState.EMPTY:
                self.__board_numpy[:, coord // self.__size, coord % self.__size] = 0.0

    def do_pass(self):
        """何もせずに手番を入れ替える"""
        self.__side_to_move = self.opponent_color

    def update(self, coord: int):
        """指定した座標に手番の石を置き，局面を更新する

        Args:
            coord (int): 手を打つ座標

        Raises:
            ValueError: その座標に手を打てない場合
        """
        assert(self.__winner == IntersectionState.EMPTY)
        assert(self.get_intersection_state_at(coord) == IntersectionState.EMPTY)
        
        pad_coord = self.__TO_PADDED_COORD[coord]
        self.__board[pad_coord] = self.__side_to_move
        self.__empties ^= 1 << coord
        self.__empty_count -= 1
        if self.__board_numpy is not None:
            self.__board_numpy[self.__side_to_move, coord // self.__size, coord % self.__size] = 1.0

        # 4方向に対して連続した石の数を数える
        for dir in (1, self.__padded_size, self.__padded_size + 1, self.__padded_size - 1):
            count = 1
            for i in range(1, Position.STONE_COUNT_TO_WIN):
                if self.__board[pad_coord + dir * i] != self.__side_to_move:
                    break
                count += 1

            for i in range(1, Position.STONE_COUNT_TO_WIN):
                if self.__board[pad_coord - dir * i] != self.__side_to_move:
                    break
                count += 1

            if count >= Position.STONE_COUNT_TO_WIN:
                self.__winner = self.__side_to_move
                break
        
        self.__side_to_move = self.opponent_color

    def undo(self, coord: int):
        """指定した座標の手を取り消し，局面を更新する.

        Args:
            coord (int): 取り消す手の座標. 直前に打たれた手である必要がある.
        """
        assert(self.get_intersection_state_at(coord) != IntersectionState.EMPTY)

        self.__side_to_move = self.opponent_color
        pad_coord = self.__TO_PADDED_COORD[coord]
        self.__board[pad_coord] = IntersectionState.EMPTY
        self.__empties |= 1 << coord
        self.__empty_count += 1
        self.__winner = IntersectionState.EMPTY
        if self.__board_numpy is not None:
            self.__board_numpy[self.__side_to_move, coord // self.__size, coord % self.__size] = 0.0

    def enumerate_empties(self):
        """空き交点を列挙する"""
        if self.__empty_count == 0:
            return

        empties = self.__empties
        coord = (empties & -empties).bit_length() - 1   # 最下位ビットの位置を取得するイディオム
        empties &= empties - 1  # 最下位ビットをクリアするイディオム
        yield coord

        while empties != 0:
            coord = (empties & -empties).bit_length() - 1
            yield coord
            empties &= empties - 1

    def enumerate_non_empties(self):
        """空き交点以外の交点を列挙する"""
        if self.__empty_count == self.__size ** 2:
            return

        non_empties = ~self.__empties & ((1 << (self.__size ** 2)) - 1)  
        coord = (non_empties & -non_empties).bit_length() - 1   # 最下位ビットの位置を取得するイディオム
        non_empties &= non_empties - 1  # 最下位ビットをクリアするイディオム
        yield coord

        while non_empties != 0:
            coord = (non_empties & -non_empties).bit_length() - 1
            yield coord
            non_empties &= non_empties - 1

    def check_winner(self) -> IntersectionState:
        """
        全ての交点を確認して勝利条件を満たしているかを確認する．

        重い処理なので，通常は局面更新時(update)に勝者を確認する．
        """
        if self.__winner != IntersectionState.EMPTY:
            return self.__winner
        
        for coord in range(self.__size ** 2):
            state = self.get_intersection_state_at(coord)
            if state == IntersectionState.EMPTY:
                continue
            
            for dir in (1, self.__padded_size, self.__padded_size + 1, self.__padded_size - 1):
                count = 1
                for i in range(1, Position.STONE_COUNT_TO_WIN):
                    if self.get_intersection_state_at(coord + dir * i) != state:
                        break
                    count += 1

                for i in range(1, Position.STONE_COUNT_TO_WIN):
                    if self.get_intersection_state_at(coord - dir * i) != state:
                        break
                    count += 1

                if count >= Position.STONE_COUNT_TO_WIN:
                    self.winner = state
                    return state
                
    def calc_manhattan_distance(self, coord_0: int, coord_1: int) -> int:
        """coord1とcoord2のマンハッタン距離を計算する"""
        return self.__MANHATTAN_TABLE[coord_0][coord_1]
    
    def to_text_line(self) -> str:
        line = []
        for i in range(self.__size):
            for j in range(self.__size):
                state = self.get_intersection_state_at(i * self.__size + j)
                if state == IntersectionState.BLACK:
                    line.append('X')
                elif state == IntersectionState.WHITE:
                    line.append('O')
                else:
                    line.append('-')

        line.append(f" {'X' if self.__side_to_move == IntersectionState.BLACK else 'O'}")

        return ''.join(line)

    def __str__(self):
        lines = []
        for i in range(self.__size):
            row = []
            for j in range(self.__size):
                state = self.get_intersection_state_at(i * self.__size + j)
                if state == IntersectionState.BLACK:
                    row.append('X')
                elif state == IntersectionState.WHITE:
                    row.append('O')
                else:
                    row.append('.')
            lines.append(' '.join(row))

        return '\n'.join(lines)

Position.static_init()
