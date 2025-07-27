from abc import ABC, abstractmethod

import gomoku
from gomoku import Position, IntersectionState
from error import GTPError

class Engine(ABC):
    """思考エンジン(AI)の抽象基底クラス"""

    def __init__(self, name: str, version: str):
        super().__init__()
        self.__name = name
        self.__version = version
        self._pos = Position(9)
        self._move_history: list[(IntersectionState, int)] = []
    
    @property
    def name(self) -> str:
        """エンジンの名前"""
        return self.__name

    @property
    def version(self) -> str:
        """エンジンのバージョン"""
        return self.__version

    @property
    def board_size(self) -> int:
        """盤面のサイズ"""
        return self._pos.size

    def quit(self):
        """エンジンを終了する"""
        pass

    def clear_board(self):
        """盤面をクリアする"""
        self._pos = Position(self.board_size)
        self._move_history.clear()

    def get_color_at(self, coord: int) -> IntersectionState:
        """指定した座標の交点の状態を取得する"""
        return self._pos.get_intersection_state_at(coord)
    
    def play(self, color: IntersectionState, coord: int) -> bool:
        """指定した座標に手を打つ"""
        if self._pos.winner != IntersectionState.EMPTY \
            or self._pos.empty_count == 0 \
            or self.get_color_at(coord) != IntersectionState.EMPTY:
            return False
        
        if coord == gomoku.PASS_COORD:
            self._pos.do_pass()
            return True
        
        if color != self._pos.side_to_move:
            self._pos.do_pass()

        self._pos.update(coord)
        self._move_history.append((color, coord))
        return True
    
    def show_board(self) -> str:
        """現在の盤面を文字列で返す"""
        return f"\n{str(self._pos)}"
    
    def undo(self) -> bool:
        """直前の手を取り消す"""
        if not self._move_history:
            return False
        
        color, last_move = self._move_history.pop()

        if color != self._pos.opponent_color:
            self._pos.do_pass()

        self._pos.undo(last_move)
        return True
    
    def get_final_result(self) -> str:
        if self._pos.empty_count == 0 and self._pos.winner == IntersectionState.EMPTY:
            return "Draw"
        elif self._pos.winner == IntersectionState.BLACK:
            return "Black wins."
        elif self._pos.winner == IntersectionState.WHITE:
            return "White wins."
        return ""
    
    def get_legal_moves(self) -> list[int]:
        """現在の盤面における合法手を取得する"""
        return list(self._pos.enumerate_empties())
    
    def get_side_to_move(self) -> IntersectionState:
        """現在の手番の色を取得する"""
        return self._pos.side_to_move
    
    def get_original_commands(self) -> list[str]:
        """エンジンがもつ固有のコマンドを取得する"""
        return []
    
    def exec_original_command(self, cmd: str, args: list[str]) -> str:
        """エンジン固有のコマンドを実行する"""
        raise GTPError(f"Unknown command: {cmd}")
    
    def set_board_size(self, size: int):
        """盤面のサイズを設定する""" 
        if size < Position.BOARD_SIZE_MIN or size > Position.BOARD_SIZE_MAX:
            return False
        
        self._pos = Position(size)
        self._move_history.clear()
        return True

    @abstractmethod
    def gen_move(self) -> int:
        """次の手を決定する"""
        pass

    def set_time(self, main_time: int, byoyomi: int, byoyomi_stones: int):
        """
        思考時間を設定する

        GoGuiからはカナダ形式の時間設定が送られてくる
        """
        pass

    def set_time_left(self, color: IntersectionState, time_left: int, byoyomi_stones_left: int):
        """
        残り時間を設定する

        GoGuiとエンジンの間での時間の同期をとるために使用される
        """
        pass