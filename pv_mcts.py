"""
APV-MCTS (asynchronous policy value MCTS)の実装
"""
import time 
import math
import random
import gc
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from gomoku import Position, IntersectionState
from dual_net import DualNet


class Network:
    def __init__(self, path: str, board_size: int, device: torch.device):
        self.__model = DualNet(board_size)
        self.__model.load_state_dict(torch.load(path))
        self.__model = self.__model.to(device)
        self.__model.eval()
        self.__device = device

    def predict(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            batch = batch.to(self.__device)
            p_logit, v_logit = self.__model(batch)
            v = torch.nn.functional.sigmoid(v_logit).cpu().numpy()
        return p_logit.cpu().numpy(), v
    

class Node:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.moves: list[int] = None

        # DNNが出力する方策と価値
        self.policy: np.ndarray = None
        self.predicted_value: float = None

        # 子ノード関連の情報
        self.child_visit_counts: np.ndarray = None
        self.child_value_sum: np.ndarray = None
        self.child_value: np.ndarray = None
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
        assert self.is_expanded
        self.child_nodes = [None] * len(self.moves)

    def expand(self, pos: Position):
        """子ノードを展開する

        Args:
            pos (Position): 現在の局面
        """
        # 子ノードの訪問回数と価値の合計を初期化
        # 実際にその子ノードに訪問する必要があるまでNodeオブジェクトは生成しない
        self.moves = list(pos.enumerate_empties())
        num_nodes = len(self.moves)
        self.child_visit_counts = np.zeros(num_nodes, dtype=np.int32)
        self.child_value_sum = np.zeros(num_nodes, dtype=np.float32)


@dataclass
class MoveEval:
    """
    探索の結果得られた着手の価値
    """
    move: int
    effort: float   # この着手に費やされた探索の割合
    simulation_count: int   # この着手に対するシミュレーションの回数
    prob: float    # この着手のDNNが予測した確率
    value: float    # 探索の結果得られたこの着手の価値
    pv: list[int] = None  # 読み筋(Principal Variation)


@dataclass
class SearchResult:
    """
    探索の結果
    """
    root_value: MoveEval
    move_evals: list[MoveEval]


@dataclass
class PUCTConfig:
    # DNNの出力に対するソフトマックス温度. DNNが出力する方策が決定論的すぎるときは読み抜けを防ぐために高い温度にする
    softmax_temperature: float = 1.5   

    # PUCT式におけるハイパーパラメータ
    # AlphaZeroのPUCT式を参照
    puct_base = 19652.0
    puct_init = 1.25

    # まとめて評価する局面数
    batch_size = 32

    # root_dirichlet_alpha = 0.03  # ルートノードのディリクレノイズのalpha
    # root_exploration_fraction = 0.25    # ルートノードのディリクレノイズの割合

@dataclass
class TrajectoryItem:
    node: Node
    child_idx: int


class VisitResult(Enum):
    """
    ルートノードに訪問した結果を表す列挙型
    """
    
    # 未評価ノードに達したため, 探索経路をキューに追加した
    QUEUEING = 0

    # 既に評価待ちになっているノードに再訪問したため, 探索経路を破棄
    DISCARDED = 1

    # 勝敗が決しているノードに達した
    TERMINAL = 2


class Searcher:
    """
    探索を行うクラス
    """

    """
    ルートノード直下のノードのFPU(First Play Urgency)
    FPUは未訪問ノードの行動価値. ルートノード直下以外の子ノードは, 親ノードの価値をFPUとして用いる.

    Note:
        ルートノード直下の未訪問ノードは全て勝ちと見做す. そうすれば, 1手先の全ての子ノードは初期に少なくとも1回は訪問される.

        ルートノード直下以外の未訪問ノードは, 親ノードの価値で初期化する. 
        そうすれば, 親ノードの価値以上の子ノードがしばらく選ばれ続ける.
    """
    __ROOT_FPU = 1.0

    """
    評価待ちノードに与えるペナルティ.
    ここでは, 評価待ちノードを1回分の負けと見做す.
    評価待ちノードとそこに至る際に経由するノードの価値を下げることで, 評価待ちノードへの再訪問を抑える.
    """
    __VIRTUAL_LOSS = 1

    def __init__(self, config: PUCTConfig, network: Network):
        self.__SOFTMAX_TEMPERATURE_INV = 1/ config.softmax_temperature
        self.__C_BASE = config.puct_base
        self.__C_INIT = config.puct_init
        self.__BATCH_SIZE = config.batch_size
        self.__network = network

        self.__simulation_count = 0
        self.__search_start_ms = 0
        self.__search_end_ms = 0
        self.__eval_count = 0

        self.__root_pos: Position = None
        self.__root: Node = None
        self.__predict_queue: list[Node] = []
        self.__batch: list[torch.Tensor] = []

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
    
    def set_root_pos(self, pos: Position):
        """探索のルート局面を設定する

        Args:
            pos (Position): ルート局面
        """
        self.__root_pos = Position(pos.size, nn_input=True)
        pos.copy_to(self.__root_pos)
        self.__root = Node()
        self.__init_root_child_nodes()
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
        
        for m, node in zip(self.__root.moves, self.__root.child_nodes):
            if m == move:
                if node is None or not node.is_expanded:
                    return False
                
                self.__root_pos.update(move)
                self.__root = node

        self.__init_root_child_nodes()
        gc.collect()

        return True
    
    def search(self, max_simulation_count: int, time_limit_ms: int) -> SearchResult:
        root_pos = self.__root_pos
        pos = self.__root_pos.copy()
        trajectories: list[list[TrajectoryItem]] = []
        trajectories_discarded: list[list[TrajectoryItem]] = []
        current_time_ms = self.__search_start_ms = int(time.perf_counter() * 1000)
        self.__simulation_count = 0
        self.__eval_count = 0

        stop = lambda: self.__simulation_count >= max_simulation_count or (current_time_ms - self.__search_start_ms) >= time_limit_ms

        while not stop():
            trajectories.clear()
            trajectories_discarded.clear()
            self.__clear_queue()

            for _ in range(self.__BATCH_SIZE):
                root_pos.copy_to(pos)
                trajectories.append([])
                result = self.__visit_root_node(pos, trajectories[-1])

                if result != VisitResult.DISCARDED:
                    self.__simulation_count += 1
                else:
                    trajectories_discarded.append(trajectories[-1])

                    if len(trajectories_discarded) > self.__BATCH_SIZE >> 1:
                        trajectories.pop()
                        break

                if result != VisitResult.QUEUEING:
                    trajectories.pop()

            if len(trajectories) > 0:
                self.__predict()

            for trajectory in trajectories_discarded:
                # 評価待ちノードにvirtual lossがダブルカウントされているので除去する
                self.__remove_virtual_loss(trajectory)

            for trajectory in trajectories:
                # 評価結果を探索経路に伝播する
                self.__backup(trajectory)

            current_time_ms = int(time.perf_counter() * 1000)

        self.__search_end_ms = current_time_ms
        return self.__collect_search_result()

    def __collect_search_result(self) -> SearchResult:
        root = self.__root
        root_value = None
        move_evals = []

        if root.visit_count > 0:
            root_value = MoveEval(
                move=-1,
                effort=1.0,
                simulation_count=root.visit_count,
                prob=1.0,
                value= root.value
            )

        if root.is_expanded:
            for i, move in enumerate(root.moves):
                value = root.child_value_sum[i] / root.child_visit_counts[i] if root.child_visit_counts[i] > 0 else 0.0
                effort = root.child_visit_counts[i] / root.visit_count 
                move_eval = MoveEval(
                    move=move,
                    effort=effort,
                    simulation_count=root.child_visit_counts[i],
                    prob=root.policy[i],
                    value=value
                )
                move_eval.pv = []
                if root.child_nodes[i] is not None:
                    self.__probe_pv(root.child_nodes[i], move_eval.pv)
                else:
                    move_eval.pv.append(move)
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

    def __init_root_child_nodes(self):
        pos = self.__root_pos
        root = self.__root

        if not root.is_expanded:
            root.expand(pos)

        if root.child_nodes is None:
            root.init_child_nodes()

        if root.policy is None or root.predicted_value is None:
            pos_tensor = pos.to_tensor()
            pos_tensor = pos_tensor.unsqueeze(0)
            p_logit, value = self.__network.predict(pos_tensor)

            root.policy = self.__softmax(p_logit[0][root.moves])
            root.predicted_value = float(value[0])

    def __visit_root_node(self, pos: Position, trajectory: list[TrajectoryItem]) -> VisitResult | float:
        """
        ルートノードを訪問する.
        このメソッドを起点として葉ノードまで降る.
        """
        node = self.__root
        child_idx = self.__select_root_child_node()
        move = node.moves[child_idx]
        node.visit_count += Searcher.__VIRTUAL_LOSS
        node.child_visit_counts[child_idx] += Searcher.__VIRTUAL_LOSS

        current_side = pos.side_to_move
        pos.update(move)

        if pos.is_gameover:
            # 勝敗が決したら, 直ちにその結果を返す.
            outcome = 0.0
            if pos.winner == IntersectionState.EMPTY:
                outcome = 0.5
            else:
                outcome = 1.0 if pos.winner == current_side else 0.0

            self.__update_stats(node, child_idx, outcome)

            return 1.0 - outcome

        trajectory.append(TrajectoryItem(node, child_idx))

        if node.child_nodes[child_idx] is None:
            # 初訪問
            child_node = node.child_nodes[child_idx] = Node()
            child_node.expand(pos)
            self.__enqueue_node(pos, child_node)
            return VisitResult.QUEUEING
        elif node.child_nodes[child_idx].predicted_value is None:
            # 訪問済みだが評価待ち
            return VisitResult.DISCARDED
        
        result = self.__visit_node(pos, node.child_nodes[child_idx], trajectory)

        if isinstance(result, VisitResult):
            return result

        self.__update_stats(node, child_idx, result)
        return 1.0 - result

    def __visit_node(self, pos: Position, node: Node, trajectory: list[TrajectoryItem]) -> VisitResult | float:
        """
        ノードを訪問し, 必要に応じて評価待ちキューに追加する

        Args:
            pos (Position): 現在の局面
            node (Node): 訪問するノード
            trajectory (list[TrajectoryItem]): 探索経路

        Returns:
            VisitResult | float: 訪問結果, もしくはnodeの価値
        """
        if node.child_nodes is None:
            node.init_child_nodes()
            
        child_idx = self.__select_child_node(node)
        move = node.moves[child_idx]

        node.visit_count += Searcher.__VIRTUAL_LOSS
        node.child_visit_counts[child_idx] += Searcher.__VIRTUAL_LOSS

        current_side = pos.side_to_move
        pos.update(move)

        if pos.is_gameover:
            # 勝敗が決したら, 直ちにその結果を返す.
            outcome = 0.0
            if pos.winner == IntersectionState.EMPTY:
                outcome = 0.5
            else:
                outcome = 1.0 if pos.winner == current_side else 0.0

            self.__update_stats(node, child_idx, outcome)

            return 1.0 - outcome
        
        trajectory.append(TrajectoryItem(node, child_idx))

        if node.child_nodes[child_idx] is None:
            # 初訪問
            child_node = node.child_nodes[child_idx] = Node()
            child_node.expand(pos)
            self.__enqueue_node(pos, child_node)
            return VisitResult.QUEUEING
        elif node.child_nodes[child_idx].predicted_value is None:
            # 訪問済みだが評価待ち
            return VisitResult.DISCARDED
        
        result = self.__visit_node(pos, node.child_nodes[child_idx], trajectory)

        if isinstance(result, VisitResult):
            return result
        
        self.__update_stats(node, child_idx, result)
        return 1.0 - result  
            

    def __select_root_child_node(self) -> int:
        parent = self.__root

        # 行動価値の計算
        # ただし，未訪問子ノードはFPUで初期化.
        q = np.divide(parent.child_value_sum, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, self.__ROOT_FPU, np.double), where=parent.child_visit_counts != 0)
        
        # バイアス項の計算
        if parent.visit_count == 0:
            # どの子ノードも訪問されていない場合はpolicyに従わせるため, バイアスは1とする.
            u = 1.0
        else:
            # PUCT式に基づいたバイアス項の計算
            sqrt_sum = math.sqrt(parent.visit_count)
            u = sqrt_sum / (1.0 + parent.child_visit_counts)

        # 探索と活用のバランスを決める係数を算出
        # AlphaZeronに基づく.
        c_base = self.__C_BASE
        c_init = self.__C_INIT
        c = math.log((1.0 + parent.visit_count + c_base) / c_base) + c_init
        return np.argmax(q + c * parent.policy * u)
    
    def __select_child_node(self, parent: Node) -> int:
        # 行動価値の計算
        # ただし，未訪問子ノードは親ノードの価値で初期化.
        fpu = parent.value if parent.visit_count > 0 else parent.predicted_value
        q = np.divide(parent.child_value_sum, parent.child_visit_counts,
                      out=np.full(parent.num_child_nodes, fpu, np.double), where=parent.child_visit_counts != 0)

        if parent.visit_count == 0:
            u = 1.0
        else:
            sqrt_sum = math.sqrt(parent.visit_count)
            u = sqrt_sum / (1.0 + parent.child_visit_counts)

        c_base = self.__C_BASE
        c_init = self.__C_INIT
        c = math.log((1.0 + parent.visit_count + c_base) / c_base) + c_init
        return np.argmax(q + c * parent.policy * u)
    
    def __enqueue_node(self, pos: Position, node: Node):
        """
        評価待ちキューにノードを追加する

        Args:
            pos (Position): 現在の局面
            node (Node): 評価待ちノード
        """
        self.__batch.append(pos.to_tensor())
        self.__predict_queue.append(node)

    def __clear_queue(self):
        """評価待ちキューをクリアする"""
        self.__predict_queue.clear()
        self.__batch.clear()
    
    def __predict(self):
        """
        キューに溜まった評価待ちノードをDNNで評価する
        """
        batch = torch.stack(self.__batch)
        p_logits, values = self.__network.predict(batch)
        
        for i, node in enumerate(self.__predict_queue):
            node.policy = self.__softmax(p_logits[i][node.moves])
            node.predicted_value = float(values[i])
        
        self.__eval_count += len(self.__predict_queue)
        self.__predict_queue.clear()
        self.__batch.clear()

    def __update_stats(self, parent: Node, child_idx: int, value: float):
        """
        親ノードの統計情報をvirtual lossを取り除きながら更新する

        Args:
            parent (Node): 親ノード
            child_idx (int): 子ノードのインデックス
            value (float): 子ノードの価値
        """
        parent.visit_count += 1 - Searcher.__VIRTUAL_LOSS
        parent.value_sum += value
        parent.child_visit_counts[child_idx] += 1 - Searcher.__VIRTUAL_LOSS
        parent.child_value_sum[child_idx] += value

    def __remove_virtual_loss(self, trajectory: list[TrajectoryItem]):
        """
        探索経路のvirtual lossを取り除く
        """
        for item in trajectory:
            item.node.visit_count -= Searcher.__VIRTUAL_LOSS
            item.node.child_visit_counts[item.child_idx] -= Searcher.__VIRTUAL_LOSS

    def __backup(self, trajectory: list[TrajectoryItem]):
        """
        経路に沿って, ルートノードに向かって価値を伝播する
        """
        value = None
        for item in reversed(trajectory):
            node = item.node
            child_idx = item.child_idx

            if value is None:
                value = 1.0 - node.child_nodes[child_idx].predicted_value

            self.__update_stats(node, child_idx, value)
            value = 1.0 - value

    def __softmax(self, x: np.ndarray) -> np.ndarray:
        beta = self.__SOFTMAX_TEMPERATURE_INV
        x = x * beta
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x)