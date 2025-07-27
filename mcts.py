""""
モンテカルロ木探索(MCTS)の実装

アルゴリズムにはUCT(Upper Confidence Bound for Trees)を用いる．
UCTでは, ノード選択にUCB1を, 局面評価には一様ランダムな方策によるシミュレーションを用いる．
"""

import math
import random
import array
import gc
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from gomoku import Position, IntersectionState


class EdgeLabel(IntEnum):
    WIN = 1
    LOSS = 0
    INTERMIDIATE = -1


class Node:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.moves: list[int] = None

        # 子ノード関連の情報
        self.edge_labels: array.array = None
        self.child_visit_counts: np.ndarray = None
        self.child_value_sums: np.ndarray = None
        self.child_nodes: list[Node] = None

    @property
    def is_expanded(self) -> bool:
        """子ノードが展開されているか"""
        return self.child_nodes is not None
    
    @property
    def num_child_nodes(self) -> int:
        """子ノードの数"""
        return len(self.child_nodes) if self.is_expanded else 0
    
    @property
    def value(self) -> float:
        """ノードの価値"""
        assert self.visit_count > 0
        return self.value_sum / self.visit_count
    
    def init_child_nodes(self, moves):
        """
        子ノードのリストを初期化する.
        """
        # ただちにNodeオブジェクトは作らずにNoneで初期化.
        # 必要になってからNodeオブジェクトを生成することでメモリ使用量を抑える.
        assert self.moves is not None
        self.child_nodes = [None] * len(self.moves)

    def expand(self, pos: Position, prev_move: int, prev_prev_move: int):
        """子ノードを展開する

        Args:
            pos (Position): 現在の局面
            prev_move (int): 直前の手
            prev_prev_move (int): 2手前の手
        """
        self.moves = list(pos.enumerate_empties())

        # 効率よく探索するために，直前の自分，もしくは相手の手の近くを優先的に探索する
        self.moves.sort(key=lambda move: min(pos.calc_manhattan_distance(move, prev_move), pos.calc_manhattan_distance(move, prev_prev_move)))

        # 子ノードの訪問回数と価値の合計を初期化
        # 実際にその子ノードに訪問する必要があるまでNodeオブジェクトは生成しない
        num_nodes = len(self.moves)
        self.edge_labels = array.array('b', [EdgeLabel.INTERMIDIATE] * num_nodes)
        self.child_visit_counts = np.zeros(num_nodes, dtype=np.int32)
        self.child_value_sums = np.zeros(num_nodes, dtype=np.double)   


@dataclass
class MoveEval:
    """
    探索の結果得られた着手の価値
    """
    move: int   
    effort: float   # この着手に費やされた探索の割合
    simulation_count: int   # この着手に対するシミュレーションの回数
    value: float    # この着手の価値


@dataclass
class SearchResult:
    """
    探索の結果
    """
    root_value: MoveEval = MoveEval()   # ルート局面の価値
    move_evals: list[MoveEval] = [] # 各着手の価値


@dataclass
class UCTConfig:
    expansion_threshold: int = 20  # 子ノードを展開閾値
    ucb_factor: float = math.sqrt(2.0)  # UCB1の係数（デフォルトは理論値のsqrt(2)）
    reuse_subtree: bool = True  # 可能なら前回の探索結果を利用する


class Searcher:
    """
    探索を行うクラス
    """

    """
    ルートノード直下のノードのFPU(First Play Urgency)
    FPUは未訪問ノードの行動価値. ルートノード直下以外の子ノードは, 親ノードの価値をFPUとして用いる.

    Note:
        ルートノード直下の未訪問ノードは全て勝ちと見做す. そうすれば, 1手先の全ての子ノードは初期に少なくとも1回はプレイアウトされる.

        ルートノード直下以外の未訪問ノードは, 親ノードの価値で初期化する. 
        そうすれば, 親ノードよりも価値の高い子ノードが見つかれば, しばらくそのノードが選ばれ続ける.
    """
    __ROOT_FPU = 1.0

    def __init__(self, config: UCTConfig):
        self.__EXPAND_THRES = config.expansion_threshold
        self.__C_UCB = config.ucb_factor
        self.__REUSE_SUBTREE = config.reuse_subtree

        self.__simulation_count = 0
        self.__search_start_ms = 0
        self.__search_end_ms = 0

        self.__root_pos: Position = None
        self.__root: Node = None

    @property
    def elapsed_ms(self) -> int:
        """探索にかかった時間(ミリ秒)"""
        return self.__search_end_ms - self.__search_start_ms
    
    @property
    def simulation_count(self) -> int:
        """探索におけるシミュレーションの回数"""
        return self.__simulation_count
    
    @property
    def pps(self) -> float:
        """探索の速度(1秒あたりのシミュレーション回数)"""
        elapsed_sec = self.elapsed_ms * 10**-3  
        return self.simulation_count / elapsed_sec if elapsed_sec > 0 else 0.0
    
    def set_root_pos(self, pos: Position, prev_move: int, prev_prev_move: int):
        """探索のルート局面を設定する

        Args:
            pos (Position): ルート局面
            prev_move (int): 直前の手
            prev_prev_move (int): 2手前の手
        """
        prev_root_pos = self.__root_pos
        self.__root_pos = pos.copy()

        


    

