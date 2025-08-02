import sys
import math

import torch
import numpy as np

import gomoku
from gomoku import Position
from engine import Engine
from error import GTPError
from dual_net import DualNet


class DualNetEngine(Engine):
    def __init__(self, board_size: int, model_path: str):
        super().__init__("PV Network Engine", "1.0")
        self.__model = DualNet(board_size)
        self.__board_size = board_size

        self.__model.load_state_dict(torch.load(model_path, map_location="cpu"))

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

        with torch.no_grad():
            self.__model.eval()
            p, v = self.__model(pos_tensor)
            p = torch.softmax(p, dim=1)
            v = torch.sigmoid(v).squeeze(1)

        legal_moves = list(self._pos.enumerate_empties())
        move_evals = []
        policy_sum = 0.0
        for move in legal_moves:
            policy = p[0, move].item()
            move_evals.append((move, policy))
            policy_sum += policy

        for i in range(len(move_evals)):
            move, policy = move_evals[i]
            if policy_sum > 0:
                move_evals[i] = (move, policy / policy_sum)
            else:
                move_evals[i] = (move, 1.0 / len(move_evals))
        move_evals.sort(key=lambda x: x[1], reverse=True)

        print(f"win_rate: {v.item() * 100.0:.2f}%", file=sys.stderr)
        self.__print_move_evals(move_evals[:5])

        return max(legal_moves, key=lambda a: p[0, a].item())

    def __print_move_evals(self, move_evals):
        for move, policy in move_evals:
            print(f"Move: {self.__coord_to_str(move)}, Policy: {policy * 100.0:.2f}%", file=sys.stderr)

    def __coord_to_str(self, coord: int) -> str:
        if coord == gomoku.PASS_COORD:
            return "pass"
        
        x = coord % self.board_size
        y = coord // self.board_size

        x_chr = chr(ord('A') + x)
        if x_chr >= 'I':
            x_chr = chr(ord(x_chr) + 1)

        return f"{x_chr}{self.board_size - y}"


if __name__ == "__main__":
    from gtp import GTP
    from server_client import ServerClient

    protocol = "gtp"
    if len(sys.argv) > 1:
        protocol = sys.argv[1]

    engine = DualNetEngine(9, "params\\DualNet\\dualnet.pth")
    if protocol == "gtp":
        gtp = GTP(engine)
        gtp.mainloop("gtp.log")
    elif protocol == "server":
        server_client = ServerClient(engine)
        server_client.mainloop("server.log")