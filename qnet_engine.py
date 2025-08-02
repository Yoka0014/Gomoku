import sys

import gomoku
from gomoku import Position
from engine import Engine
from error import GTPError
from dqn_train import DQNConfig, Networks, outcome_to_win

class QNetworkEngine(Engine):
    def __init__(self, config: DQNConfig, model_path: str):
        super().__init__("Q Network Engine", "1.0")
        self.__network = Networks(config.board_size, config.value_transform, model_path)
        self.__board_size = config.board_size

    def set_board_size(self, size):
        if size != self.__board_size:
            raise GTPError(f"Board size {size} is not supported. Current board size is {self.__board_size}.")
        return super().set_board_size(size)
    
    def gen_move(self):
        if self._pos.winner != gomoku.IntersectionState.EMPTY or self._pos.empty_count == 0:
            raise GTPError("Game is already over.")

        if self._pos.empty_count == 1:
            return self._pos.enumerate_empties().__next__()
        
        pos = Position(self._pos.size, nn_input=True)
        self._pos.copy_to(pos)
        pos_tensor = pos.to_tensor()
        pos_tensor = pos_tensor.unsqueeze(0)
        q = self.__network.predict_q(pos_tensor).detach().cpu().numpy()[0]
        legal_moves = list(self._pos.enumerate_empties())
        move_evals = [(a, q[a]) for a in legal_moves]
        move_evals.sort(key=lambda x: x[1], reverse=True)

        self.__print_move_evals(move_evals[:5])

        return move_evals[0][0]

    def __print_move_evals(self, move_evals):
        for move, eval in move_evals:
            print(f"Move: {move}, WinRate: {outcome_to_win(eval) * 100.0:.2f}%", file=sys.stderr)


if __name__ == "__main__":
    from gtp import GTP
    from server_client import ServerClient

    protocol = "gtp"
    if len(sys.argv) > 1:
        protocol = sys.argv[1]

    engine = QNetworkEngine(DQNConfig(), "dqn_model_1999.pth")
    if protocol == "gtp":
        gtp = GTP(engine)
        gtp.mainloop("gtp.log")
    elif protocol == "server":
        server_client = ServerClient(engine)
        server_client.mainloop("server.log")