import sys

import gomoku
from engine import Engine
from error import GTPError
from mcts import UCTConfig, Searcher, SearchResult

class UCTEngine(Engine):
    def __init__(self):
        super().__init__("UCT Engine", "1.0")
        self.__searcher = Searcher(UCTConfig())
        self.__simulatioin_count = 100000
        self.__time_limit_ms = 10**8
        self.__num_moves_to_print = 5   # ログに上位何手までを表示するか

        self.__original_commands ={
            "set_simulations": self.__exec_set_simulations_command,
            "set_time_limit_per_move": self.__exec_set_time_limit_per_move_command,
            "set_num_moves_to_print": self.__exec_set_num_moves_to_print_command,
         }
        
        self.__init_root()

    def clear_board(self):
        super().clear_board()
        self.__searcher.set_root_pos(self._pos)

    def play(self, color, coord):
        if not super().play(color, coord):
            return False

        if self.__searcher.try_update_root_pos(coord):
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
        
        if self._pos.empty_count == self._pos.size ** 2:
            return self._pos.size ** 2 // 2  # 初手天元
        
        result = self.__searcher.search(self.__simulatioin_count, self.__time_limit_ms)

        # GTPの通信と競合するので，探索結果は標準エラー出力に出力する.
        print(self.__search_result_to_str(result), file=sys.stderr)

        move_evals = result.move_evals
        return max(move_evals, key=lambda x: x.effort).move

    def __init_root(self):
        if self._pos.empty_count == 0:
            return

        move_history = list(filter(lambda x: x[1] != gomoku.PASS_COORD, self._move_history))
        if len(move_history) == 0:
            self.__searcher.set_root_pos(self._pos)
        elif len(move_history) == 1:
            self.__searcher.set_root_pos(self._pos, move_history[0][1])
        else:
            self.__searcher.set_root_pos(self._pos, move_history[-1][1], move_history[-2][1])

    def __exec_set_simulations_command(self, args: list[str]) -> str:
        if len(args) != 1:
            raise GTPError("Invalid number of arguments for set_simulations command.")
        
        try:
            self.__simulatioin_count = int(args[0])
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
    
    def __search_result_to_str(self, res: SearchResult) -> str:
        s = []
        s.append(f"elapsed={self.__searcher.elapsed_ms}[ms]\t{self.__searcher.simulation_count}[simulations]\t{self.__searcher.pps}[pps]\n")
        s.append(f"win_rate={res.root_value.value * 100:.2f}%\n")
        s.append("|move|effort|playouts|win_rate|\n")

        for eval in sorted(res.move_evals, key=lambda x: x.effort, reverse=True)[:min(self.__num_moves_to_print, len(res.move_evals))]:
            s.append(f"|{self.__coord_to_str(eval.move).rjust(4)}|")
            s.append(f"{eval.effort * 100.0:.2f}%".rjust(6))
            s.append('|')
            s.append(f"{eval.simulation_count:>8}|")
            s.append(f"{eval.value * 100:.2f}%".rjust(8))
            s.append("|\n")

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