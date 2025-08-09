"""
DQNのトレーニングを行うスクリプト.

手法には, Dueling DQNを採用し, 行動価値更新式はDouble DQNを用いる.
"""
import math
import random
from collections import deque
from typing import Iterator

import numpy as np

import torch
import torch.nn as nn

from gomoku import Position, IntersectionState
from dual_net import DualNet


OUTCOME_WIN = 1
OUTCOME_LOSS = -1
OUTCOME_DRAW = 0

def outcome_to_win(outcome: float) -> float:
    return (outcome + 1) / 2


class DQNConfig:
    """
    学習全般の設定
    """
    def __init__(self):
        self.initial_model_path = None  # 初期モデルのパス, Noneならば乱数で初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.board_size = 15
        
        self.learning_rate = 0.001
        
        # DNNが出力する価値を変換する関数
        # デフォルトはtanh関数を用い, 勝ち, 負け, 引き分けをそれぞれ1, -1, 0とする.
        self.value_transform = nn.Tanh()

        # 価値を相手の視点に変換する関数
        self.negate_value = lambda v: -v

        self.loss_function = nn.HuberLoss()

        self.batch_size = 256

        # ターゲットネットワークの更新頻度
        # ネットワークパラメータがtarget_net_update_freq回更新されるごとにターゲットネットワークを更新する
        self.target_net_update_freq = 5

        # ネットワークパラメータの更新回数がtrain_stepsに達したら学習を終了する
        self.train_steps = 7 * 10**5

        # ReplayBufferの容量. 単位はエピソード.
        # 例えば, 30000ならば, 直近30000エピソード分の経験を保持する.
        #
        # あまり大きくすると学習が遅くなるが，小さすぎても局所解に陥りやすくなる.
        # 9路盤では30000程度で十分学習が進むが, 15路盤では10倍ほど増やさないと局所解に陥る．
        self.replay_buffer_capacity = 3 * 10**5

        # ReplayBufferにエピソードがwarmup_sizeだけ溜まるまで学習を行わない.
        self.warmup_size = self.batch_size * 100

        # epsilon-greedy方策のパラメータ
        self.epsilon_start = 0.9    # epsilonの初期値
        self.epsilon_end = 0.05   # epsilonの最小値
        self.epsilon_decay = 2000  # epsilonの減衰率, 大きくすればするほど減衰速度が遅くなる

        self.model_path = "dqn_model_15_{0}.pth"  
        self.loss_history_path = "loss_history.txt"
        self.win_rate_history_path = "win_rate_history.txt"
        self.save_model_freq = 1000  # モデルを保存する頻度.

    def init_pos(self) -> Position:
        """
        初期局面を生成する.
        デフォルトはPositionクラスのコンストラクタ.

        この関数を書き換えることで任意局面を初期局面として設定できる.
        """
        return Position(self.board_size, nn_input=True)


class Episode:
    """
    1エピソード分の情報を保持するクラス, すなわち棋譜
    """
    def __init__(self, root_pos: Position):
        self.__root_pos = root_pos.copy()
        self.__move_history: list[int] = []

    def __len__(self):
        return len(self.__move_history)
    
    def add_move(self, move: int):
        self.__move_history.append(move)

    def sample(self) -> tuple[Position, int]:
        """
        エピソードからランダムに1局面をサンプリングする.
        戻り値は, (局面, 着手)のタプル.
        """
        idx = random.randint(0, len(self.__move_history) - 1)
        pos = self.__root_pos.copy()
        for move in self.__move_history[:idx]:
            pos.update(move)
        return pos, self.__move_history[idx]
    

class ReplayBuffer:
    """
    経験再生用のバッファ.

    いわゆるdequeであり, 直近capacityエピソードを保持する.
    """
    def __init__(self, capacity: int):
        self.__episodes: deque[Episode] = deque(maxlen=capacity)

    @property
    def window_size(self) -> int:
        return self.__episodes.maxlen
    
    def __len__(self):
        return len(self.__episodes)
    
    def add_episode(self, episode: Episode):
        self.__episodes.append(episode)

    def sample_batch(self, batch_size: int) -> list[tuple[Position, int, Position, float|None]]:
        """
        バッファからランダムにbatch_size個の局面をサンプリングする.
        戻り値は, (局面, 着手, 次の局面, 報酬)のタプルのリスト.
        報酬は, 終局した局面以外ではNoneとなり, 次の局面のプレイヤからみた報酬を意味する.

        過学習防止のため, 同じエピソードから複数の局面がサンプリングされることはないようにする.
        """
        # エピソードのサンプリングは, その長さに比例した確率で行う.
        episodes_len_sum = sum(len(ep) for ep in self.__episodes)
        episodes = np.random.choice(list(self.__episodes), size=batch_size, p=[len(ep)/episodes_len_sum for ep in self.__episodes], replace=False)

        batch = []
        for pos, move in [ep.sample() for ep in episodes]:
            pos: Position

            next_pos = pos.copy()
            reward = None
            next_pos.update(move)
            if next_pos.is_gameover:
                if next_pos.winner == IntersectionState.EMPTY:
                    reward = OUTCOME_DRAW
                else:
                    reward = OUTCOME_WIN if next_pos.side_to_move == next_pos.winner else OUTCOME_LOSS

            batch.append((pos, move, next_pos, reward))

        return batch
    

class Networks:
    """
    DNNを管理するクラス.

    DQNでは, 学習対象のネットワークとターゲットネットワークの両方を使い分けるため, 一箇所で管理する.
    """
    def __init__(self, board_size, value_transform=None, path=None):
        if path is None:
            self.__qnet = DualNet(board_size)
            self.__qnet.init_weights()
        else:
            self.__qnet = DualNet(board_size)
            self.__qnet.load_state_dict(torch.load(path, map_location=torch.device("cpu"), weights_only=True))

        self.__target_qnet = DualNet(board_size)
        self.__target_qnet.load_state_dict(self.__qnet.state_dict())

        self.value_transform = value_transform

        self.__step_count = 0

    @property
    def step_count(self) -> int:
        return self.__step_count
    
    def save_at(self, path: str):
        torch.save(self.__qnet.state_dict(), path)
    
    def set_device(self, device: torch.device):
        self.__qnet.to(device)
        self.__target_qnet.to(device)
    
    def get_parameters(self) -> Iterator[nn.Parameter]:
        return self.__qnet.parameters()

    def update_target_network(self):
        self.__target_qnet.load_state_dict(self.__qnet.state_dict())

    def predict_q(self, batch: torch.Tensor) -> torch.Tensor:
        """
        局面のバッチを入力として, 各局面における行動価値Q(s, a)を出力する.
        """
        self.__qnet.eval()
        with torch.no_grad():
            q = self.__predict_q(self.__qnet, batch.to(next(self.__qnet.parameters()).device))
        self.__qnet.train()
        return q

    def step(self, batch: list[tuple[Position, int, Position, float|None]], optimizer: torch.optim.Optimizer, loss_func, negate_value) -> float:
        """
        局面, 着手, 次の局面, 報酬のバッチからパラメータを更新する.
        """
        qnet, target_net = self.__qnet, self.__target_qnet

        device = next(qnet.parameters()).device
        pos_batch = torch.stack([pos.to_tensor() for pos, _, _, _ in batch]).to(device)
        next_pos_batch = torch.stack([next_pos.to_tensor() for _, _, next_pos, _ in batch]).to(device)

        with torch.no_grad():
            qnet.eval()
            q_next = self.__predict_q(qnet, next_pos_batch)
            qnet.train()

            target_net.eval()
            target_q = self.__predict_q(target_net, next_pos_batch)

        td_targets = []
        for i, (_, _, next_pos, reward) in enumerate(batch):
            if reward is not None:
                td_targets.append(negate_value(reward))
                continue

            legal_moves = list(next_pos.enumerate_empties())

            assert len(legal_moves) > 0 # 報酬がNoneのときは必ず合法手があるはず.

            # Double DQNでは, 学習中のネットワークで価値が最大の行動(best_move)を選択.
            best_move = max(legal_moves, key=lambda a: q_next[i, a])

            # 次にbest_moveの価値はターゲットネットワークから取得.
            # 最大値を計算するネットワークと価値を計算するネットワークを分けることで最大化バイアスを抑える.
            td_targets.append(negate_value(target_q[i, best_move]))

        td_targets = torch.tensor(td_targets, dtype=torch.float32).to(device)
        q = self.__predict_q(qnet, pos_batch)
    
        # 実際に選択した行動の価値のみを集める
        actions = torch.tensor([move for _, move, _, _ in batch], dtype=torch.int64).to(device)
        q = q.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        loss = loss_func(q, td_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.__step_count += 1

        return loss.item()


    def __predict_q(self, model, batch: torch.Tensor) -> torch.Tensor:
        adv, v = model(batch)
        q = v + (adv - adv.mean(dim=1, keepdim=True))

        if self.value_transform is not None:
            q = self.value_transform(q)

        return q
    

def epsilon(config: DQNConfig, step_count: int):
    e_start = config.epsilon_start
    e_end = config.epsilon_end
    e_decay = config.epsilon_decay
    return e_end + (e_start - e_end) * math.exp(-step_count / e_decay)


def exec_episodes(config: DQNConfig, networks: Networks, replay_buffer: ReplayBuffer) -> float:
    """
    エピソードをbatch_size分だけ並行に実行し, 結果をReplayBufferに追加する.
    また, 全エピソードにおける黒側から見た勝率を返す.
    """
    positions = [config.init_pos() for _ in range(config.batch_size)]
    episodes = [Episode(pos) for pos in positions]

    while any(not pos.is_gameover for pos in positions):
        q = networks.predict_q(torch.stack([pos.to_tensor() for pos in positions])).detach().cpu().numpy()
        moves = select_moves(networks.step_count, positions, q)

        for episode, pos, move in zip(episodes, positions, moves):
            if not pos.is_gameover:
                pos.update(move)
                episode.add_move(move)

    for episode in episodes:
        replay_buffer.add_episode(episode)

    outcomes = [OUTCOME_DRAW if pos.winner == IntersectionState.EMPTY else OUTCOME_WIN if pos.winner == IntersectionState.BLACK else OUTCOME_LOSS for pos in positions]
    return sum([outcome_to_win(o) for o in outcomes]) / len(outcomes)


def select_moves(step_count:int, positions: list[Position], q: np.ndarray) -> list[int]:
    """
    各局面において, epsilon-greedy方策に従って行動を選択する.
    """
    moves = []
    for i, pos in enumerate(positions):
        if pos.is_gameover:
            moves.append(-1)
            continue

        if random.random() < epsilon(config, step_count):
            legal_moves = list(pos.enumerate_empties())
            move = random.choice(legal_moves)
        else:
            move = max(pos.enumerate_empties(), key=lambda a: q[i, a])
        moves.append(move)
    return moves


if __name__ == "__main__":
    config = DQNConfig()
    networks = Networks(config.board_size, value_transform=config.value_transform, path=config.initial_model_path)
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)
    optimizer = torch.optim.Adam(networks.get_parameters(), lr=config.learning_rate)

    networks.set_device(config.device)

    episode_count = 0
    while networks.step_count < config.train_steps:
        print(f"Step {networks.step_count}, Episode {episode_count + 1} to {episode_count + config.batch_size}, Epsilon: {epsilon(config, networks.step_count):.4f}")

        win_rate = exec_episodes(config, networks, replay_buffer)
        episode_count += config.batch_size

        print(f"Win rate from black: {win_rate * 100.0:.2f}%")

        with open(config.win_rate_history_path, "a") as f:
            f.write(f"({networks.step_count},{win_rate})\n")

        if (networks.step_count + 1) % config.target_net_update_freq == 0:
            networks.update_target_network()
            print("Target network has been updated.")

        if (networks.step_count + 1) % config.save_model_freq == 0:
            path = config.model_path.format(networks.step_count)
            networks.save_at(path)
            print(f"Model has been saved at \"{config.model_path.format(networks.step_count)}\"")

        if len(replay_buffer) >= config.warmup_size:
            batch = replay_buffer.sample_batch(config.batch_size)
            loss = networks.step(batch, optimizer, config.loss_function, config.negate_value)
            print(f"Loss: {loss:.4f}")

            with open(config.loss_history_path, "a") as f:
                f.write(f"({networks.step_count},{loss})\n")

        print()

    path = config.model_path.format("final")
    networks.save_at(path)

    
