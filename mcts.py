""""
モンテカルロ木探索(MCTS)の実装

アルゴリズムにはUCT(Upper Confidence Bound for Trees)を用いる．
UCTでは, ノード選択にUCB1を, 局面評価には一様ランダムな方策によるシミュレーションを用いる．
"""
import time
import math
import random
import gc
from dataclasses import dataclass

import numpy as np

from gomoku import Position, IntersectionState


class Node:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.moves: list[int] = None

        # 子ノード関連の情報
        self.child_visit_counts: np.ndarray = None
        self.child_value_sums: np.ndarray = None
        self.child_nodes: list[Node] = None

    @property
    def is_expanded(self) -> bool:
        """子ノードが展開されているか"""
        return self.moves is not None
    
    @property
    def num_child_nodes(self) -> int:
        """子ノードの数"""
        return len(self.moves)

    @property
    def value(self) -> float:
        """ノードの価値"""
        assert self.visit_count > 0
        return self.value_sum / self.visit_count
    
    def init_child_nodes(self):
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
    pv: list[int] = None  # 読み筋(Principal Variation)


@dataclass
class SearchResult:
    """
    探索の結果
    """
    root_value: MoveEval
    move_evals: list[MoveEval]


@dataclass
class UCTConfig:
    expansion_threshold: int = 1  # 子ノードを展開閾値
    ucb_factor: float = 0.5  # UCB1の係数（デフォルトは理論値のsqrt(2)）


class Searcher:
    """
    探索を行うクラス
    """

    """
    ルートノード直下のノードのFPU(First Play Urgency)
    FPUは未訪問ノードの行動価値.

    Note:
        ルートノード直下の未訪問ノードは全て勝ちと見做す. そうすれば, 1手先の全ての子ノードは初期に少なくとも1回は訪問される.

        ルートノード直下以外の未訪問ノードは, 負けで初期化する. 
        そうすれば, 最初に選ばれたノードがしばらく選ばれ続ける．
    """
    __ROOT_FPU = 1.0
    __FPU = 0.0

    def __init__(self, config: UCTConfig):
        self.__EXPAND_THRES = config.expansion_threshold
        self.__C_UCB = config.ucb_factor

        self.__simulation_count = 0
        self.__search_start_ms = 0
        self.__search_end_ms = 0

        self.__root_pos: Position = None
        self.__root_prev_move: int = None
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
    
    def set_root_pos(self, pos: Position, prev_move: int=-1, prev_prev_move: int=-1):
        """探索のルート局面を設定する

        Args:
            pos (Position): ルート局面
            prev_move (int): 直前の手
            prev_prev_move (int): 2手前の手
        """
        # 以前の着手位置が不明な場合は中央の座標を使用する
        center_coord = pos.size // 2 + (pos.size // 2) * pos.size
        if prev_move == -1:
            prev_move = center_coord

        if prev_prev_move == -1:
            prev_prev_move = center_coord

        self.__root_pos = pos.copy()
        self.__root_prev_move = prev_move
        self.__root = Node()
        self.__init_root_child_node(prev_move, prev_prev_move)
        gc.collect()

    def try_update_root_pos(self, move: int) -> bool:
        """ルート局面を更新し, 前回の探索結果を再利用する

        Args:
            move (int): ルート局面に適用する着手

        Returns:
            bool: ルート局面が更新されたかどうか
        """
        if self.__root_pos is None:
            return False

        if not self.__root.is_expanded or move not in self.__root.moves or self.__root.child_nodes is None:
            return False

        prev_prev_move = self.__root_prev_move
        for m, node in zip(self.__root.moves, self.__root.child_nodes):
            if m == move:
                if node is None or not node.is_expanded:
                    return False
                
                self.__root_pos.update(move)
                self.__root_prev_move = move
                self.__root = node

        self.__init_root_child_node(self.__root_prev_move, prev_prev_move)
        gc.collect()

        return True
        
    def search(self, max_simulation_count: int, time_limit_ms: int) -> SearchResult:
        root_pos = self.__root_pos
        pos = root_pos.copy()
        self.__search_start_ms = int(time.perf_counter() * 1000)
        self.__simulation_count = 0
        for _ in range(max_simulation_count):
            current_time_ms = int(time.perf_counter() * 1000)
            if time_limit_ms > 0 and current_time_ms - self.__search_start_ms >= time_limit_ms:
                break

            root_pos.copy_to(pos)
            self.__visit_root_node(pos)
            self.__simulation_count += 1

        self.__search_end_ms = int(time.perf_counter() * 1000)
        return self.__collect_search_result()

    def __collect_search_result(self) -> SearchResult:
        root = self.__root
        move_evals = []

        if root.visit_count > 0:
            root_value = MoveEval(
                move=-1,
                effort=1.0,
                simulation_count=root.visit_count,
                value=root.value_sum / root.visit_count
            )

        if root.is_expanded:
            for i, move in enumerate(root.moves):
                if root.child_nodes[i] is None:
                    continue

                value = root.child_value_sums[i] / root.child_visit_counts[i] if root.child_visit_counts[i] > 0 else 0.0
                effort = root.child_visit_counts[i] / root.visit_count
                move_eval = MoveEval(move, effort, root.child_visit_counts[i], value, [move])
                self.__probe_pv(root.child_nodes[i], move_eval.pv)
                move_evals.append(move_eval)

        return SearchResult(root_value, move_evals)
    
    def __probe_pv(self, node: Node, pv: list[int]):
        """PV(Principal Variation)を探す

            PVとは読み筋のこと
        """
        if not node.is_expanded or node.visit_count < 100:
            return 

        best_idx = 0
        for i in range(node.num_child_nodes):
            if node.child_visit_counts[i] > node.child_visit_counts[best_idx]:
                best_idx = i
        pv.append(node.moves[best_idx])

        if node.child_nodes is not None and node.child_nodes[best_idx] is not None:
            self.__probe_pv(node.child_nodes[best_idx], pv)

    def __init_root_child_node(self, prev_move: int, prev_prev_move: int):
        pos: Position = self.__root_pos
        root: Node = self.__root

        if not root.is_expanded:
            root.expand(pos, prev_move, prev_prev_move)

        if root.child_nodes is None:
            root.init_child_nodes()

        for i in range(root.num_child_nodes):
            if root.child_nodes[i] is None:
                root.child_nodes[i] = Node()

    def __visit_root_node(self, pos: Position):
        node = self.__root
        child_idx = self.__select_root_child_node()
        move = node.moves[child_idx]
        pos.update(move)

        if pos.winner != IntersectionState.EMPTY:
            value = 0.0 if pos.winner == pos.side_to_move else 1.0
        else:
            value = 1 - self.__visit_node(pos, node.child_nodes[child_idx], move, self.__root_prev_move)

        self.__update_stats(node, child_idx, value)

    def __visit_node(self, pos: Position, node: Node, prev_move: int, prev_prev_move: int) -> float:
        if not node.is_expanded:
            node.expand(pos, prev_move, prev_prev_move)

        if node.num_child_nodes == 1:
            # 勝敗を決定できる.
            pos.update(node.moves[0])
            if pos.winner == IntersectionState.EMPTY:
                return 0.5
            
            # 勝敗は反転して返す
            return 0.0 if pos.winner == pos.side_to_move else 1.0

        child_idx = self.__select_child_node(node)
        move = node.moves[child_idx]
        pos.update(move)

        if pos.winner != IntersectionState.EMPTY:
            value = 0.0 if pos.winner == pos.side_to_move else 1.0
            self.__update_stats(node, child_idx, value)
            return value

        if node.child_visit_counts[child_idx] < self.__EXPAND_THRES:
            # 子ノードが展開閾値に達していない場合はrolloutを行う
            value = 1.0 - self.__rollout(pos)
            self.__update_stats(node, child_idx, value)
            return value
        
        if node.child_nodes is None:
            node.init_child_nodes()
        
        if node.child_nodes[child_idx] is None:
            node.child_nodes[child_idx] = Node()

        child_node = node.child_nodes[child_idx]
        value = 1.0 - self.__visit_node(pos, child_node, move, prev_move)
        self.__update_stats(node, child_idx, value)
        return value
        
    def __select_root_child_node(self) -> int:
        """UCB1に基づいてルートノードの子ノードを選択する"""
        parent = self.__root
        visit_sum = parent.visit_count
        not_visit_count = np.sum(parent.child_visit_counts == 0)
        log_sum = math.log(visit_sum + not_visit_count)

        # 行動価値の計算
        # ただし，未訪問子ノードはFPUで初期化.
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, self.__ROOT_FPU, np.double), where=parent.child_visit_counts != 0)
        
        # バイアス項の計算
        u = np.divide(log_sum, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, log_sum, np.double), where=parent.child_visit_counts != 0)
        np.sqrt(u, out=u)

        return np.argmax(q + self.__C_UCB * u)

    def __select_child_node(self, parent: Node) -> int:
        visit_sum = parent.visit_count
        not_visit_count = np.sum(parent.child_visit_counts == 0)
        log_sum = math.log(visit_sum + not_visit_count)

        # 行動価値の計算
        # ただし，未訪問子ノードはFPUで初期化.
        q = np.divide(parent.child_value_sums, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, self.__FPU, np.double), where=parent.child_visit_counts != 0)

        # バイアス項の計算
        u = np.divide(log_sum, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, log_sum, np.double), where=parent.child_visit_counts != 0)
        np.sqrt(u, out=u)

        return np.argmax(q + self.__C_UCB * u)
    
    def __rollout(self, pos: Position) -> float:
        """ランダムロールアウトを行い，局面の価値を推定する

        元々, この処理のことをMCTSではplayoutと呼ぶのが一般的だが,
        対局シミュレーションによる価値の推定は, 強化学習におけるrolloutと同じなので
        ここではrolloutと呼ぶことにする.

        Args:
            pos (Position): 現在の局面

        Returns:
            float: 勝敗(win=1.0, loss=0.0, draw=0.5)
        """
        root_side = pos.side_to_move
        empties = list(pos.enumerate_empties())
        random.shuffle(empties)

        for move in empties:
            pos.update(move)
            if pos.winner != IntersectionState.EMPTY:
                break

        if pos.winner == IntersectionState.EMPTY:
            return 0.5

        return 1.0 if pos.winner == root_side else 0.0

    def __update_stats(self, parent: Node, child_idx: int, value: float):
        """ノードの統計情報を更新する"""
        parent.visit_count += 1
        parent.value_sum += value
        parent.child_visit_counts[child_idx] += 1
        parent.child_value_sums[child_idx] += value
