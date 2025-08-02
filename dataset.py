import torch

from gomoku import Position, IntersectionState


class PositionDataset:
    """
    (局面, 着手, 勝敗)のデータセット
    """

    def __init__(self, path: str, max_count = 1.0 * 10 ** 18, verbose: bool = True):
        # メモリを節約するために, 局面はbitboardで保持する.
        self.__positions: list[tuple[int, int, int]] = []
        self.__moves: list[int] = []
        self.__outcomes: list[float] = []

        with open(path, "r") as file:
            for i, line in enumerate(file):
                if i >= max_count:
                    break

                pos, move, outcome = PositionDataset.parse_dataset_text(line)
                self.__positions.append(pos)
                self.__moves.append(move)
                self.__outcomes.append(outcome)

                if i % 100000 == 0 and verbose:
                    print(f"loaded {i} positions")

    def __len__(self) -> int:
        return len(self.__positions)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        index番目の局面, 着手, 勝敗を返す.
        """
        pos = self.__positions[index]
        move = self.__moves[index]
        outcome = self.__outcomes[index]

        pos_tensor = PositionDataset.__position_to_tensor(pos)
        move_tensor = torch.tensor(move, dtype=torch.int64)
        outcome_tensor = torch.tensor(outcome, dtype=torch.float32)

        return pos_tensor, move_tensor, outcome_tensor

    @staticmethod
    def __position_to_tensor(pos: tuple[int, int, int]) -> torch.Tensor:
        """
        局面をone-hotベクトルに変換する.
        posは(盤面サイズ, プレイヤーのbitboard, 相手のbitboard)のタプル.
        """
        board_size, player, opponent = pos
        tensor = torch.zeros((2, board_size, board_size), dtype=torch.float32)
        for coord in range(board_size ** 2):
            if player & (1 << coord):
                tensor[0, coord // board_size, coord % board_size] = 1.0
            elif opponent & (1 << coord):
                tensor[1, coord // board_size, coord % board_size] = 1.0

        return tensor

    @staticmethod
    def position_to_text(pos: Position, move: int, outcome: float) -> str:
        """
        データセットでは, Xを手番側の石, Oを相手側の石, -を空きマスとして表現する.
        また, 最後にその局面の直後に行われた着手と対局の結果を付与する.
        """
        text = []
        for coord in range(pos.size ** 2):
            if pos.get_intersection_state_at(coord) == pos.side_to_move:
                text.append('X')
            elif pos.get_intersection_state_at(coord) == pos.opponent_color:
                text.append('O')
            else:
                text.append('-')

        text.append(f" {move} {outcome}")

        return ''.join(text)
    
    @staticmethod
    def parse_dataset_text(text: str) -> tuple[tuple[int, int, int], int, float]:
        """
        データセットのテキスト表現から局面, 着手, 勝敗を抽出する.
        """
        board, move, outcome = text.strip().split()
        board_size = Position.SQRT_TABLE[len(board)]

        player = 0
        opponent = 0

        for coord, state in enumerate(board):
            if state == 'X':
                player |= 1 << coord
            elif state == 'O':
                opponent |= 1 << coord

        move = int(move)
        outcome = float(outcome)

        return (board_size, player, opponent), move, outcome
