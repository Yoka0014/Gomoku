"""GTP (Go Text Protocol) に準拠したインターフェース．
GoGui(https://github.com/Remi-Coulom/gogui?tab=readme-ov-file)との通信で用いる．
"""
import datetime

import gomoku
from gomoku import Position, IntersectionState
from engine import Engine
from error import GTPError

class GTP:
    VERSION = "2.0"

    def __init__(self, engine: Engine):
        self.__engine = engine
        self.__commands = {}
        self.__quit_flag = False
        self.__log_write = lambda x: None
        self.__log_write_line = lambda x: self.__log_write(f"{x}\n")
        self.__init_commands()

    def __init_commands(self):
        self.__commands = {
            "protocol_version": self.__exec_protocol_version_command,
            "name": self.__exec_name_command,
            "version": self.__exec_version_command,
            "known_command": self.__exec_know_command_command,
            "list_commands": self.__exec_list_commands_command,
            "quit": self.__exec_quit_command,
            "boardsize": self.__exec_board_size_command,
            "clear_board": self.__exec_clear_board_command,
            "komi": self.__exec_komi_command,
            "play": self.__exec_play_command,
            "genmove": self.__exec_gen_move_command,
            "reg_genmove": self.__exec_reg_gen_move_command,
            "undo": self.__exec_undo_command,
            "showboard": self.__exec_showboard_command,
            "time_settings": self.__exec_time_settings_command,
            "time_left": self.__exec_time_left_command,
            "loadsgf": self.load_sgf_command,
            "color": self.__exec_color_command,

            # gogui-rule commands
            "gogui-rules_game_id": self.__exec_rules_game_id_command,
            "gogui-rules_board": self.__exec_showboard_command,
            "gogui-rules_board_size": self.__exec_rules_board_size_command,
            "gogui-rules_legal_moves": self.__exec_rules_legal_moves_command,
            "gogui-rules_side_to_move": self.__exec_rules_side_to_move_command,
            "gogui-rules_final_result": self.__exec_rules_final_result_command,
        }

    def mainloop(self, log_file_path: str = None):
        """コマンド受付のメインループ．

        Args:
            log_file_path (str, optional): ログファイルのパス. デフォルトはNoneで、ログは出力しない.
        """
        logger = None
        if log_file_path:
            logger = open(log_file_path, "w", encoding="utf-8")

            def write(s: str):
                logger.write(s)
                logger.flush()

            self.__log_write = write

        while not self.__quit_flag:
            cin = input()

            self.__log_write_line(f"[{datetime.datetime.now()}] Input: {cin}")

            cmd_name, args, id = self.__parse_command(cin)

            try:
                if self.__commands.get(cmd_name, None):
                    self.__commands[cmd_name](id, args)
                else:
                    if cmd_name in self.__engine.get_original_commands():
                        self.success(id, self.__engine.exec_original_command(cmd_name, args))
                    else:
                        self.failure(id, f"unknown command: {cmd_name}")
            except GTPError as e:
                self.failure(id, e.message)

        if log_file_path:
            logger.close()

        self.__quit_flag = False

    @staticmethod
    def __parse_command(cmd: str) -> tuple[str, list[str], int | None]:
        """コマンドを分解して、(コマンド名, 引数リスト, ID)のタプルを返す
        """
        splitted = cmd.lower().split()
        has_id = cmd.isdigit()
        offset = 1 if has_id else 0

        args = []
        for i in range(offset + 1, len(splitted)):
            args.append(splitted[i])

        return (splitted[offset], args, int(splitted[0]) if has_id else None)

    def success(self, id: int | None, msg: str):
        id_str = str(id) if id is not None else ""
        output = f"={id_str} {msg}\n"

        self.__log_write_line(f"[{datetime.datetime.now()}] Status: Success Output: {output}")

        print(output, end="\n\n", flush=True)

    def failure(self, id: int | None, msg: str):
        id_str = str(id) if id is not None else ""
        output = f"{id_str} {msg}\n"

        self.__log_write_line(f"[{datetime.datetime.now()}] Status: Failure Output: {output}")

        print(f"?{output}", end="\n\n", flush=True)

    def __exec_protocol_version_command(self, id: int | None, args: list[str]):
        self.success(id, self.VERSION)

    def __exec_name_command(self, id: int | None, args: list[str]):
        self.success(id, self.__engine.name)

    def __exec_version_command(self, id: int | None, args: list[str]):
        self.success(id, self.__engine.version)

    def __exec_know_command_command(self, id: int | None, args: list[str]):
        """指定されたコマンドが存在するかどうかを確認するコマンド"""
        if len(args) == 0:
            self.failure(id, "invalid option")
            return
        
        self.success(id, "true" if args[0] in self.__commands else "false")

    def __exec_list_commands_command(self, id: int | None, args: list[str]):
        """サポートされているコマンドのリストを返す"""
        commands = self.__commands.keys()
        self.success(id, "\n".join(commands))

    def __exec_quit_command(self, id: int | None, args: list[str]):
        self.__engine.quit()
        self.__quit_flag = True
        self.success(id, "")

    def __exec_board_size_command(self, id: int | None, args: list[str]):
        """盤面のサイズを設定するするコマンド"""
        
        if len(args) == 0:
            self.failure(id, "invalid option")
            return
        
        if not args[0].isdigit():
            self.failure(id, "board size must be a positive integer")
            return
        
        if self.__engine.set_board_size(int(args[0])):
            self.success(id, "")
        else:
            self.failure(id, "unacceptable size")

    def __exec_clear_board_command(self, id: int | None, args: list[str]):
        """盤面をクリアするコマンド"""
        self.__engine.clear_board()
        self.success(id, "")

    def __exec_komi_command(self, id: int | None, args: list[str]):
        """
        コミを設定するコマンド

        コミは囲碁特有のルールであるため，特に何もしない
        """
        if len(args) == 0:
            self.failure(id, "invalid option")
            return
        
        self.success(id, "")

    def __exec_play_command(self, id: int | None, args: list[str]):
        """指定された座標に石を打つコマンド"""
        if len(args) < 2:
            self.failure(id, "invalid option")
            return
        
        color = self.__parse_color(args[0])
        if color == IntersectionState.EMPTY:
            self.failure(id, "invalid color")
            return
        
        coord = self.__parse_coord(args[1])
        if coord == -1:
            self.failure(id, "invalid coordinate")
            return
        
        if not self.__engine.play(color, coord):
            self.failure(id, "invalid move")
            return

        self.success(id, "")

    def __exec_gen_move_command(self, id: int | None, args: list[str]):
        """次の手を生成して着手するコマンド"""
        if len(args) == 0:
            self.failure(id, "invalid option")
            return
        
        color = self.__parse_color(args[0])
        if color != self.__engine.get_side_to_move():
            self.failure(id, "invalid color")
            return
        
        coord = self.__engine.gen_move()
        self.__engine.play(color, coord)
        self.success(id, self.__coord_to_str(coord))

    def __exec_reg_gen_move_command(self, id: int | None, args: list[str]):
        """次の手を生成するが着手はしないコマンド"""
        if len(args) == 0:
            self.failure(id, "invalid option")
            return
        
        color = self.__parse_color(args[0])
        if color == IntersectionState.EMPTY:
            self.failure(id, "invalid color")
            return
        
        coord = self.__engine.gen_move(color)
        self.success(id, self.__coord_to_str(coord))

    def __exec_undo_command(self, id: int | None, args: list[str]):
        """直前の着手を取り消すコマンド"""
        if self.__engine.undo():
            self.success(id, "")
        else:
            self.failure(id, "cannot undo")

    def __exec_showboard_command(self, id: int | None, args: list[str]):
        """現在の盤面を表示するコマンド"""
        self.success(id, self.__engine.show_board())

    def __exec_time_settings_command(self, id: int | None, args: list[str]):
        """時間設定コマンド"""
        if len(args) < 3:
            self.failure(id, "invalid option")
            return
        
        if not args[0].isdigit() or not args[1].isdigit() or not args[2].isdigit():
            self.failure(id, "time settings must be integers")
            return
        
        self.__engine.set_time(int(args[0]), int(args[1]), int(args[2]))
        self.success(id, "")

    def __exec_time_left_command(self, id: int | None, args: list[str]):
        """残り時間を設定するコマンド"""
        if len(args) < 3:
            self.failure(id, "invalid option")
            return
        
        color = self.__parse_color(args[0])
        if color == IntersectionState.EMPTY:
            self.failure(id, "invalid color")
            return
        
        if not args[1].isdigit() or not args[2].isdigit():
            self.failure(id, "time left must be integers")
            return
        
        self.__engine.set_time_left(color, int(args[1]), int(args[2]))
        self.success(id, "")

    def load_sgf_command(self, id: int | None, args: list[str]):
        """
        SGFファイルを読み込むコマンド

        現状は非対応
        """
        self.failure(id, "SGF file is not supported")

    def __exec_color_command(self, id: int | None, args: list[str]):
        """指定された座標の石の色を取得するコマンド"""
        if len(args) < 1:
            self.failure(id, "invalid option")
            
        coord = self.__parse_coord(args[0])
        if coord == -1:
            self.failure(id, "invalid coordinate")
            return
        
        color = self.__engine.get_color_at(coord)
        if color == IntersectionState.EMPTY:
            self.success(id, "empty")
        elif color == IntersectionState.BLACK:
            self.success(id, "black")
        elif color == IntersectionState.WHITE:
            self.success(id, "white")
        else:
            assert False, "Unknown color state"

    def exec_showboard_command(self, id: int | None, args: list[str]):
        """現在の盤面を表示するコマンド"""
        self.success(id, self.__engine.show_board())

    """
    gogui-rule commands

    GoGuiにgomokuのルールを適用するためのコマンド
    """

    def __exec_rules_game_id_command(self, id: int | None, args: list[str]):
       self.success(id, "gomoku")

    def __exec_rules_board_size_command(self, id: int | None, args: list[str]):
        self.success(id, str(self.__engine.board_size))

    def __exec_rules_legal_moves_command(self, id: int | None, args: list[str]):
        """現在の盤面における合法手を取得するコマンド"""
        
        moves = []
        if self.__engine._pos.winner == IntersectionState.EMPTY and self.__engine._pos.empty_count > 0:
            moves = [self.__coord_to_str(coord) for coord in self.__engine.get_legal_moves()]

        if(len(moves) != 0):
            moves.append("pass")

        self.success(id, " ".join(moves))

    def __exec_rules_side_to_move_command(self, id: int | None, args: list[str]):
        """現在の手番の色を取得するコマンド"""
        side = self.__engine.get_side_to_move()
        if side == IntersectionState.BLACK:
            self.success(id, "black")
        elif side == IntersectionState.WHITE:
            self.success(id, "white")
        else:
            assert False, "Unknown side to move state"

    def __exec_rules_final_result_command(self, id: int | None, args: list[str]):
        """ゲームの最終結果を取得するコマンド"""
        self.success(id, self.__engine.get_final_result())
        
    @staticmethod
    def __parse_color(color: str) -> IntersectionState:
        color = color.lower()
        if color == "black" or color == "b":
            return IntersectionState.BLACK
        elif color == "white" or color == "w":
            return IntersectionState.WHITE
        else:
            return IntersectionState.EMPTY
        
    def __parse_coord(self, coord: str) -> int:
        coord = coord.lower()

        if coord == "pass":
            return gomoku.PASS_COORD

        if len(coord) < 2:
            return -1
        
        x = ord(coord[0]) - ord('a')

        # GoGuiは座標にiを用いないため調整が必要
        if coord[0] > 'i':
            x -= 1

        y = int(coord[1:])
        y = self.__engine.board_size - y
        return x + y * self.__engine.board_size
    
    def __coord_to_str(self, coord: int) -> str:
        if coord == gomoku.PASS_COORD:
            return "pass"
        
        x = coord % self.__engine.board_size
        y = coord // self.__engine.board_size

        # GoGuiは座標にiを用いない.
        x_chr = chr(ord('A') + x)
        if x_chr >= 'I':
            x_chr = chr(ord(x_chr) + 1)

        return f"{x_chr}{self.__engine.board_size - y}"