"""
対局サーバー用の通信プログラム
"""
import sys
import datetime

import gomoku
from gomoku import Position, IntersectionState
from engine import Engine

class ServerClient:
    def __init__(self, engine: Engine):
        self.__engine = engine
        self.__quit_flag = False
        self.__log_write = lambda x: None
        self.__log_write_line = lambda x: self.__log_write(f"{x}\n")

        self.__commands = {
            'quit': self.__exec_quit_command,
            'pos': self.__exec_pos_command,
            'move': self.__exec_move_command,
            'go': self.__exec_go_command,
        }

    def mainloop(self, log_file_path: str=None):
        logger = None
        if log_file_path:
            logger = open(log_file_path, 'a', encoding='utf-8')
            
            def write(s: str):
                logger.write(s)
                logger.flush()

            self.__log_write = write

        while not self.__quit_flag:
            cin = input()

            self.__log_write_line(f"[{datetime.datetime.now()}] Input: {cin}")

            splitted = cin.split()
            cmd_name, args = splitted[0], splitted[1:]

            if self.__commands.get(cmd_name, None):
                self.__commands[cmd_name](args)
            else:
                self.__log_write_line(f"Unknown command: {cmd_name}")

        if log_file_path:
            logger.close()

        self.__quit_flag = False

    def __exec_quit_command(self, args: list[str]):
        self.__engine.quit()
        self.__quit_flag = True

    def __exec_pos_command(self, args: list[str]):
        assert len(args) == 2, "Usage: pos <board> <side_to_move>"
        
        board_str = args[0]
        board_size = Position.SQRT_TABLE[len(board_str)]

        if board_size == -1 or board_size < Position.BOARD_SIZE_MIN or board_size > Position.BOARD_SIZE_MAX:
            assert False, "Invalid board size"

        self.__engine.set_board_size(board_size)
        for coord, state in enumerate(board_str):
            if state == 'X':
                self.__engine.play(IntersectionState.BLACK, coord)
            elif state == 'O':
                self.__engine.play(IntersectionState.WHITE, coord)
            else:
                assert state == '-', f"Invalid character '{state}' in board string"

        assert args[1] in ('X', 'O'), "Invalid side to move"

        side_to_move = IntersectionState.BLACK if args[1] == 'X' else IntersectionState.WHITE
        
        if self.__engine.get_side_to_move() != side_to_move:
            self.__engine.play(gomoku.to_opponent_color(side_to_move), gomoku.PASS_COORD)
        
    def __exec_move_command(self, args: list[str]):
        assert len(args) == 1, "Usage: move <coord>"
        
        coord = int(args[0])
        self.__engine.play(self.__engine.get_side_to_move(), coord)

    def __exec_go_command(self, args: list[str]):
        assert len(args) == 1, "Usage: genmove <time_limit_ms>"

        self.__engine.set_time(0, int(args[0]), 0)

        move = self.__engine.gen_move()
        self.__engine.play(self.__engine.get_side_to_move(), move)
        print(f"move {move}", file=sys.stdout)
        
