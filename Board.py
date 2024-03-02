import numpy as np
from enum import Enum
import time
from util import fixed_board

class MoveDir(Enum):
    """A class of move direction"""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    L_UP = 4
    R_UP = 5
    L_DOWN = 6
    R_DOWN = 7
    NONE = 8

    @staticmethod
    def opposite(dir: int) -> int:
        match dir:
            case MoveDir.LEFT.value: return MoveDir.RIGHT.value
            case MoveDir.L_UP.value: return MoveDir.R_DOWN.value
            case MoveDir.UP.value: return MoveDir.DOWN.value
            case MoveDir.R_UP.value: return MoveDir.L_DOWN.value
            case MoveDir.L_DOWN.value: return MoveDir.R_UP.value
            case MoveDir.DOWN.value: return MoveDir.UP.value
            case MoveDir.R_DOWN.value: return MoveDir.L_UP.value
            case MoveDir.RIGHT.value: return MoveDir.LEFT.value
            case MoveDir.NONE.value: return MoveDir.NONE.value
            case _: return MoveDir.NONE.value

    @staticmethod
    def int2str(dir: int) -> str:
        match dir:
            case MoveDir.LEFT.value: return 'L'
            case MoveDir.RIGHT.value: return 'R'
            case MoveDir.UP.value: return 'U'
            case MoveDir.DOWN.value: return 'D'
            case MoveDir.L_UP.value: return 'LU'
            case MoveDir.R_UP.value: return 'RU'
            case MoveDir.L_DOWN.value: return 'LD'
            case MoveDir.R_DOWN.value: return 'RD'
            case _: return 'NON'

    @staticmethod
    def str2int(s: str) -> int:
        if s == 'U':
            return MoveDir.UP.value
        elif s == 'D':
            return MoveDir.DOWN.value
        elif s == 'L':
            return MoveDir.LEFT.value
        elif s == 'R':
            return MoveDir.RIGHT.value
        elif s == 'LU':
            return MoveDir.L_UP.value
        elif s == 'LD':
            return MoveDir.L_DOWN.value
        elif s == 'RU':
            return MoveDir.R_UP.value
        elif s == 'RD':
            return MoveDir.R_DOWN.value
        else:
            return MoveDir.NONE.value

class Runes(Enum):
    """A class of runes"""
    WATER = 1
    FIRE = 2
    WOOD = 3
    LIGHT = 4
    DARK = 5
    HEART = 6
    
    @staticmethod
    def int2str(rune: int) -> str:
        for r in Runes:
            if r.value == rune:
                return r.name
        return 'invalid rune'
    
    @staticmethod
    def int2color_code(rune: int) -> str:
        if rune == Runes.WATER.value:
            return "\033[94m" # blue
        if rune == Runes.FIRE.value:
            return "\033[91m" # red
        if rune == Runes.WOOD.value:
            return "\033[92m" # green
        if rune == Runes.LIGHT.value:
            return "\033[93m" # yellow
        if rune == Runes.DARK.value:
            return "\033[95m" # magenta
        if rune == Runes.HEART.value:
            return "\033[0m" # white
        return 'invalid rune'


def MoveDir2str(dir: int) -> str:
    """convert MoveDir to string"""
    for d in MoveDir:
        if d.value == dir:
            return d.name
    return 'invalid direction'

def int2MoveDir(dir: int) -> MoveDir:
    """convert int to MoveDir"""
    for d in MoveDir:
        if d.value == dir:
            return d
    return MoveDir.NONE

def int2MoveDir_str(dir: int) -> str:
    """convert int to MoveDir string"""
    for d in MoveDir:
        if d.value == dir:
            return d.name
    return 'invalid direction'

class Board:
    """A class of tos board"""

    def __init__(self, num_col=6, num_row=5, num_rune=6, max_move=30, num_action=9, mode='random', extra_obs=4) -> None:
        self.num_col = num_col
        self.num_row = num_row
        self.num_rune = num_rune
        self.board_size = num_col * num_row
        self.max_move = max_move
        self.num_action = num_action
        self.mode = mode
        self.extra_obs = extra_obs
        self.board = np.zeros((num_row, num_col), dtype=int)
        self.reset()
        # self.print_attr()

    def print_attr(self):
        print(f'{self.num_col=}, {self.num_row=}, {self.num_rune=}, {self.max_move=}, {self.num_action=}, {self.mode=}, {self.extra_obs=}')
        print(f'{self.cur_x=}, {self.cur_y=}')

    def get_obs(self) -> np.ndarray:
        obs = np.reshape(self.board, (self.board_size,))
        obs = np.append(obs, self.cur_x)
        obs = np.append(obs, self.cur_y)
        obs = np.append(obs, self.prev_action)
        # print(f'pre_act={self.prev_action}, obs={obs[self.num_col*self.num_row:]}')
        if self.extra_obs > 3:
            obs = np.append(obs, self.move_count)
        obs = obs.astype(np.int8)
        return obs
    
    def set_board(self, board: np.ndarray) -> None:
        self.board = board.copy()

    def set_cur_pos(self, x: int, y: int) -> None:
        self.cur_x = x
        self.cur_y = y

    def set_by_obs(self, obs: np.ndarray) -> None:
        self.board = obs[:self.num_col*self.num_row].reshape((self.num_row, self.num_col))
        self.cur_x = obs[self.num_col*self.num_row]
        self.cur_y = obs[self.num_col*self.num_row+1]
        self.prev_action = obs[self.num_col*self.num_row+2]
        if len(obs) > self.num_col*self.num_row+3:
            self.move_count = self.max_move - obs[self.num_col*self.num_row+3]

    def reset_combo(self):
        self.combo = 0
        self.first_combo = 0
        self.totol_eliminated = 0
        # self.unchecked_col = [0, 1, 2, 3, 4, 5]

    def reset(self):
        self.reset_combo()
        self.move_count = 0
        self.cur_x = 0
        self.cur_y = 0
        self.action_invalid = False
        self.action = -1
        self.prev_action = -1
        self.terminate = False

        # if mode is fixed and fixed board exists, use fixed board
        if self.mode == 'fixed' and (self.num_col, self.num_row) in fixed_board:
            self.board = fixed_board[(self.num_col, self.num_row)].copy()
        else:
            while True:
                self.board = np.random.randint(1, 1+self.num_rune, (self.num_row, self.num_col))
                self.evaluate()
                if self.first_combo == 0:
                    break
        self.reset_combo()

    def random_board(self):
        self.reset()

    def random_pos(self):
        self.cur_x = np.random.randint(0, self.num_col)
        self.cur_y = np.random.randint(0, self.num_row)
        # print(f'random_pos: ({self.cur_x}, {self.cur_y})')
        self.move_count = 0

    def eliminate(self):
        is_fisrt = True
        # break if no runes to eliminate
        while True:
            # calculate to_eliminate
            to_eliminate = np.zeros((self.num_row, self.num_col), dtype=int)
            for x in range(self.num_col-2):
                for y in range(self.num_row):
                    color = self.board[y][x]
                    if self.board[y][x+1] == color and self.board[y][x+2] == color:
                        to_eliminate[y][x] = color
                        to_eliminate[y][x+1] = color
                        to_eliminate[y][x+2] = color
                
            for x in range(self.num_col):
                for y in range(self.num_row-2):
                    color = self.board[y][x]
                    if self.board[y+1][x] == color and self.board[y+2][x] == color:
                        to_eliminate[y][x] = color
                        to_eliminate[y+1][x] = color
                        to_eliminate[y+2][x] = color

            # if no runes to eliminate, break
            if not np.any(to_eliminate): return 0

            self.board -= to_eliminate
            last_y = 0
            target = 0
            # eliminate every runes
            while True:
                isZero = True
                # find the first rune to eliminate
                for i in range(last_y, self.num_row):
                    for j in range(self.num_col):
                        if to_eliminate[i][j] != 0:
                            isZero = False
                            last_y = i
                            idx = (i, j)
                            target = to_eliminate[i][j]
                            break
                    if not isZero: break
                if isZero: break
                if is_fisrt:
                    self.first_combo += 1
                self.combo += 1
                # dfs to eliminate
                stack = [idx]
                visited = []
                while stack:
                    idx = stack.pop()
                    if idx in visited: continue
                    visited.append(idx)
                    # check left, right, up, down
                    if idx[0] > 0:
                        if to_eliminate[idx[0]-1][idx[1]] == target:
                            stack.append((idx[0]-1, idx[1]))
                    if idx[0] < self.num_row-1:
                        if to_eliminate[idx[0]+1][idx[1]] == target:
                            stack.append((idx[0]+1, idx[1]))
                    if idx[1] > 0:
                        if to_eliminate[idx[0]][idx[1]-1] == target:
                            stack.append((idx[0], idx[1]-1))
                    if idx[1] < self.num_col-1:
                        if to_eliminate[idx[0]][idx[1]+1] == target:
                            stack.append((idx[0], idx[1]+1))

                    to_eliminate[idx[0]][idx[1]] = 0
                    self.totol_eliminated += 1
                # print(f'to_eliminate:\n{to_eliminate}')
            self.drop()
            is_fisrt = False
        
    def evaluate(self) -> None:
        """eliminate the board but not actually eliminate"""
        original_board = self.board.copy()
        self.reset_combo()
        self.eliminate()
        self.board = original_board
        # self.board, original_board = original_board, self.board
        # return original_board # return the eliminated board

    def drop(self):
        # drop runes
        for i in range(self.num_col):
            stack = []
            for j in range(self.num_row):
                if self.board[j][i] != 0:
                    stack.append(self.board[j][i])
            if len(stack) < self.num_row:
                stack = [0] * (self.num_row - len(stack)) + stack
            for j in range(self.num_row):
                self.board[j][i] = stack[j]

    def move(self, action: int):
        '''move rune to the direction of dir'''
        if self.action != -1:
            self.prev_action = self.action
        # print(f'prev/act: {int2MoveDir_str(self.prev_action)}/{int2MoveDir_str(action)}')

        self.action_invalid = False
        # if is first move, set current_pos
        if self.cur_x == -1:
            if 0 > action or action >= self.board_size:
                self.action_invalid = True
                return
            self.cur_x, self.cur_y = self.idx2pos(action)
            self.evaluate()
            return
        
        if action > self.num_action - 1: # invalid direction
            print(f'invalid action: {action}, {self.num_action=}')
            self.action_invalid = True
            # self.terminate = True
            return
        if action == MoveDir.NONE.value: # end move
            self.terminate = True
            return
        self.move_count += 1
        x, y = self.cur_x, self.cur_y
        if action == MoveDir.LEFT.value:
            if x == 0: 
                self.action_invalid = True
            else:
                self.cur_x -= 1
                self.board[y][x], self.board[y][x-1] = self.board[y][x-1], self.board[y][x]
        elif action == MoveDir.RIGHT.value:
            if x == self.num_col-1:
                self.action_invalid = True
            else:
                self.cur_x += 1
                self.board[y][x], self.board[y][x+1] = self.board[y][x+1], self.board[y][x]
        elif action == MoveDir.UP.value:
            if y == 0:
                self.action_invalid = True
            else:
                self.cur_y -= 1
                self.board[y][x], self.board[y-1][x] = self.board[y-1][x], self.board[y][x]
        elif action == MoveDir.DOWN.value:
            if y == self.num_row-1:
                self.action_invalid = True
            else:
                self.cur_y += 1
                self.board[y][x], self.board[y+1][x] = self.board[y+1][x], self.board[y][x]
        elif action == MoveDir.L_UP.value:
            if x == 0 or y == 0:
                self.action_invalid = True
            else:
                self.cur_x -= 1
                self.cur_y -= 1
                self.board[y][x], self.board[y-1][x-1] = self.board[y-1][x-1], self.board[y][x]
        elif action == MoveDir.R_UP.value:
            if x == self.num_col-1 or y == 0:
                self.action_invalid = True
            else:
                self.cur_x += 1
                self.cur_y -= 1
                self.board[y][x], self.board[y-1][x+1] = self.board[y-1][x+1], self.board[y][x]
        elif action == MoveDir.L_DOWN.value:
            if x == 0 or y == self.num_row-1:
                self.action_invalid = True
            else:
                self.cur_x -= 1
                self.cur_y += 1
                self.board[y][x], self.board[y+1][x-1] = self.board[y+1][x-1], self.board[y][x]
        elif action == MoveDir.R_DOWN.value:
            if x == self.num_col-1 or y == self.num_row-1:
                self.action_invalid = True
            else:
                self.cur_x += 1
                self.cur_y += 1
                self.board[y][x], self.board[y+1][x+1] = self.board[y+1][x+1], self.board[y][x]
        elif action == MoveDir.NONE.value:
            pass
        else:
            raise ValueError(f'Invalid action: {action}')

        if not self.action_invalid:
            self.action = action

        self.evaluate()

    def is_game_over(self):
        return self.terminate or self.move_count >= self.max_move

    def print_board(self):
        print('')
        for row in self.board:
            print('\033[0m|', end=' ')
            for rune in row:
                if rune == 0:
                    print(' ', end=' ')
                else:
                    print(f'\033[1m{Runes.int2color_code(rune)}●', end=' ')
            print('\033[0m|')
        print('')

    # copy constructor
    def copy(self):
        new_board = Board(self.num_col, self.num_row, self.num_rune, self.max_move, self.num_action, self.mode, self.extra_obs)
        new_board.num_col = self.num_col
        new_board.num_row = self.num_row
        new_board.num_rune = self.num_rune
        new_board.max_move = self.max_move
        new_board.num_action = self.num_action
        new_board.mode = self.mode
        new_board.extra_obs = self.extra_obs
        new_board.board = self.board.copy()
        new_board.cur_x = self.cur_x
        new_board.cur_y = self.cur_y
        new_board.move_count = self.move_count
        new_board.action_invalid = self.action_invalid
        new_board.action = self.action
        new_board.prev_action = self.prev_action
        new_board.terminate = self.terminate
        new_board.first_combo = self.first_combo
        new_board.combo = self.combo
        new_board.totol_eliminated = self.totol_eliminated
        return new_board
    
    def idx2pos(self, idx: int) -> tuple[int, int]:
        """convert index to x, y position"""
        return (idx % self.num_col, idx // self.num_col)

    def pos2idx(self, x: int, y: int) -> int:
        """convert position to index"""
        return y * self.num_col + x

def get_board(obs: np.ndarray, num_col: int, num_row: int, num_rune: int, max_move: int, num_action: int) -> Board:
    """get board from obs"""
    board = Board(num_col, num_row, num_rune, max_move, num_action)
    board.board = obs[:num_col*num_row].reshape((num_row, num_col))
    board.cur_x = obs[num_col*num_row]
    board.cur_y = obs[num_col*num_row+1]
    board.prev_action = obs[num_col*num_row+2]
    if len(obs) > num_col*num_row+3:
        board.move_count = max_move - obs[num_col*num_row+3]
    return board

def main():
    board = Board()
    board.random_board()
    print(board.board)
    board.print_board()

if __name__ == "__main__":
    main()

    