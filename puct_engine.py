import sys

import torch

import gomoku
from engine import Engine
from error import GTPError
from pv_mcts import PUCTConfig, Network, Searcher, SearchResult

class PUCTEngine(Engine):
    def __init__(self, default_network_path: str):
        super().__init__("PUCT Engine", "1.0")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__searcher = Searcher(PUCTConfig(), Network(default_network_path, self._pos.size, device))
        self.__simulation_count = 100000
        self.__time_limit_ms = 10000
        self.__reuse_subtree = True # ルートノードの子ノードを再利用するかどうか
        self.__num_moves_to_print = 5 # ログに上位何手までを表示するか

        self.__original_commands = {
            "set_simulations": self.__exec_set_simulations_command,
            "set_time_limit_per_move": self.__exec_set_time_limit_per_move_command,
            "set_num_moves_to_print": self.__exec_set_num_moves_to_print_command,
            "set_reuse_subtree": self.__exec_set_reuse_subtree_command
        }

        self.__init_root()

    def clear_board(self):
        super().clear_board()
        self.__searcher.set_root_pos(self._pos)

    def play(self, color, coord):
        if not super().play(color, coord):
            return False

        if self.__reuse_subtree and self.__searcher.try_update_root_pos(coord):
            return True

        self.__init_root()

        return True
    
    def undo(self):
        if not super().undo():
            return False
        self.__init_root()

    def set_board_size(self, size):
        if not super().set_board_size(size):
            return False
        
        self.__init_root()

        return True
    
    def get_original_commands(self) -> list[str]:
        return list(self.__original_commands.keys())
    
    def exec_original_command(self, cmd, args):
        if cmd in self.__original_commands:
            return self.__original_commands[cmd](args)
        else:
            raise GTPError(f"Unknown command: {cmd}")
        
    def gen_move(self) -> int:
        if self._pos.winner != gomoku.IntersectionState.EMPTY or self._pos.empty_count == 0:
            raise GTPError("Game is already over.")

        if self._pos.empty_count == 1:
            return self._pos.enumerate_empties().__next__()

        result = self.__searcher.search(self.__simulation_count, self.__time_limit_ms)

        # GTPの通信と競合するので，探索結果は標準エラー出力に出力する.
        print(self.__search_result_to_str(result), file=sys.stderr)

        move_evals = result.move_evals
        return max(move_evals, key=lambda x: x.effort).move
    
    def set_time(self, main_time: int, byoyomi: int, byoyomi_stones: int):
        self.__time_limit_ms = byoyomi
        
    def __init_root(self):
        if self._pos.empty_count == 0:
            return
        self.__searcher.set_root_pos(self._pos)

    def __exec_set_simulations_command(self, args: list[str]) -> str:
        if len(args) != 1:
            raise GTPError("Invalid number of arguments for set_simulations command.")
        
        try:
            self.__simulation_count = int(args[0])
        except ValueError:
            raise GTPError("Invalid argument for set_simulations command. Must be an integer.")

        return ""
    
    def __exec_set_time_limit_per_move_command(self, args: list[str]) -> str:
        if len(args) != 1:
            raise GTPError("Invalid number of arguments for set_time_limit_per_move command.")
        
        try:
            self.__time_limit_ms = int(args[0])
        except ValueError:
            raise GTPError("Invalid argument for set_time_limit_per_move command. Must be an integer.")

        return ""
    
    def __exec_set_num_moves_to_print_command(self, args: list[str]) -> str:
        if len(args) != 1:
            raise GTPError("Invalid number of arguments for set_num_moves_to_print command.")
        
        try:
            self.__num_moves_to_print = int(args[0])
        except ValueError:
            raise GTPError("Invalid argument for set_num_moves_to_print command. Must be an integer.")

        return ""
    
    def __exec_set_reuse_subtree_command(self, args: list[str]) -> str:
        if len(args) != 1:
            raise GTPError("Invalid number of arguments for set_reuse_subtree command.")
        
        if args[0].lower() == "true":
            self.__reuse_subtree = True
        elif args[0].lower() == "false":
            self.__reuse_subtree = False
        else:
            raise GTPError("Invalid argument for set_reuse_subtree command. Must be 'true' or 'false'.")
        
        return ""

    def __search_result_to_str(self, res: SearchResult) -> str:
        s = []
        s.append(f"elapsed={self.__searcher.elapsed_ms}[ms]\t{self.__searcher.simulation_count}[simulations]\t{self.__searcher.pps}[pps]\n")
        s.append(f"win_rate={res.root_value.value * 100:.2f}%\n")
        s.append("|move|policy|effort|playouts|win_rate|depth|pv\n")

        for eval in sorted(res.move_evals, key=lambda x: x.effort, reverse=True)[:min(self.__num_moves_to_print, len(res.move_evals))]:
            s.append(f"|{self.__coord_to_str(eval.move).rjust(4)}|")
            s.append(f"{eval.prob * 100.0:.2f}%".rjust(6))
            s.append('|')
            s.append(f"{eval.effort * 100.0:.2f}%".rjust(6))
            s.append('|')
            s.append(f"{eval.simulation_count:>8}|")
            s.append(f"{eval.value * 100:.2f}%".rjust(8))
            s.append('|')
            s.append(f"{len(eval.pv):>5}")
            s.append('|')
            s.append(" ".join(self.__coord_to_str(c) for c in eval.pv))
            s.append('\n')


        return "".join(s)
    
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

    engine = PUCTEngine("params/DualNet/dualnet.pth")
    if protocol == "gtp":
        gtp = GTP(engine)
        gtp.mainloop()
    elif protocol == "server":
        server_client = ServerClient(engine)
        server_client.mainloop()
