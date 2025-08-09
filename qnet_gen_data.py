"""
学習済みのQ Networkで訓練データを生成するスクリプト
"""
import random

import numpy as np
import torch

from gomoku import Position, IntersectionState
from dqn_train import DQNConfig, Networks, outcome_to_win
from dataset import PositionDataset

OUTCOME_WIN = 1.0
OUTCOME_LOSS = 0.0
OUTCOME_DRAW = 0.5

DQN_CONFIG = DQNConfig()

class GenerationConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_size = 9
        self.batch_size = 8192
        self.num_steps = 200
        self.out_path = "train_data.txt"
        self.model_path = "params/DQN/dqn_model_95999.pth"

        # 序盤に確率的な手を何手打つか
        self.num_stochastic_moves = 5

        # stochastic movesにバラツキを与える温度パラメータ.
        # 温度が高いほどランダム性が高くなる.
        self.temperature = 0.1 

    def init_pos(self):
        return Position(self.board_size, nn_input=True)


def negate_outcome(outcome: float) -> float:
    return 1.0 - outcome


class Episode:
    """
    1エピソード分の情報を保持するクラス, すなわち棋譜
    """
    def __init__(self, root_pos: Position):
        self.__root_pos = root_pos.copy()
        self.__move_history: list[int] = []
        self.__outcome_for_black: float | None = None

    def __len__(self):
        return len(self.__move_history)
    
    def add_move(self, move: int):
        self.__move_history.append(move)

    def set_outcome(self, color: IntersectionState, outcome: float):
        if color == IntersectionState.BLACK:
            self.__outcome_for_black = outcome
        elif color == IntersectionState.WHITE:
            self.__outcome_for_black = 1.0 - outcome
        else:
            raise ValueError("Invalid color for outcome.")

    def sample(self) -> tuple[Position, int, float]:
        """
        エピソードからランダムに1局面をサンプリングする.
        戻り値は, (局面, 着手, 勝敗)のタプル.
        """
        idx = random.randint(0, len(self.__move_history) - 1)
        pos = self.__root_pos.copy()
        for move in self.__move_history[:idx]:
            pos.update(move)
        
        if pos.side_to_move == IntersectionState.BLACK:
            outcome = self.__outcome_for_black
        else:
            outcome = negate_outcome(self.__outcome_for_black)

        return pos, self.__move_history[idx], outcome
    

def exec_episodes(config: GenerationConfig, networks: Networks) -> list[tuple[Position, int, float]]:
    """
    エピソードをbatch_size分だけ並行に実行し, 各エピソードから1局面ずつサンプリングする.
    """
    positions = [config.init_pos() for _ in range(config.batch_size)]
    episodes = [Episode(pos) for pos in positions]

    move_count = 0
    while any(not pos.is_gameover for pos in positions):
        q = networks.predict_q(torch.stack([pos.to_tensor() for pos in positions])).detach().cpu().numpy()
        moves = select_moves(config, positions, q, move_count)

        for episode, pos, move in zip(episodes, positions, moves):
            if not pos.is_gameover:
                pos.update(move)
                episode.add_move(move)

                if pos.is_gameover:
                    if pos.winner == IntersectionState.EMPTY:
                        episode.set_outcome(pos.side_to_move, OUTCOME_DRAW)
                    elif pos.winner == pos.side_to_move:
                        episode.set_outcome(pos.side_to_move, OUTCOME_WIN)
                    else:
                        episode.set_outcome(pos.side_to_move, OUTCOME_LOSS)

        move_count += 1
                    
    return [episode.sample() for episode in episodes]


def select_moves(config: GenerationConfig, positions: list[Position], q: np.ndarray, move_count) -> list[int]:
    """
    各局面において, greedy方策に従って行動を選択する.
    """
    moves = []
    for i, pos in enumerate(positions):
        if pos.is_gameover:
            moves.append(-1)
            continue

        if move_count < config.num_stochastic_moves:
            candidates = list(pos.enumerate_empties())
            p = [outcome_to_win(float(q[i, a])) ** (1.0 / config.temperature) for a in candidates]
            sum_p = sum(p)
            p = [x / sum_p for x in p]
            moves.append(int(np.random.choice(candidates, p=p)))
        else:
            move = max(pos.enumerate_empties(), key=lambda a: q[i, a])
            moves.append(move)

    return moves


if __name__ == "__main__":
    config = GenerationConfig()
    networks = Networks(config.board_size, DQN_CONFIG.value_transform, config.model_path)
    networks.set_device(config.device)
    out_file = open(config.out_path, 'a', encoding='utf-8')
    for step_count in range(config.num_steps):
        batch = exec_episodes(config, networks)

        for pos, move, outcome in batch:
            text = PositionDataset.position_to_text(pos, move, outcome)
            out_file.write(f"{text}\n")

        if step_count % 100 == 0:
            out_file.flush()

        print(f"[{step_count + 1}/{config.num_steps}]")

    out_file.close()