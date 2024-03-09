import os
import time
import psutil
import numpy as np

from MoveDir import MoveDir
from Runes import Runes

# from typing import Any

PPO_PATH = "ppo_model"
DQN_PATH = "dqn_model"

fixed_board: dict[tuple[int, int], np.ndarray] = {
    (3, 3): np.array(
        [
            [3, 3, 1],
            [2, 1, 2],
            [3, 2, 1]
        ]
    ),
    (4, 4): np.array(
        [
            [3, 3, 1, 1],
            [2, 1, 2, 2],
            [3, 1, 2, 1],
            [3, 2, 1, 1]
        ]
    ),
    (5, 4): np.array(
        [
            [4, 3, 1, 1, 4],
            [2, 1, 2, 2, 3],
            [3, 1, 2, 1, 4],
            [3, 2, 1, 1, 4]
        ]
    ),
    (5, 5): np.array(
        [
            [2, 6, 3, 1, 1],
            [6, 3, 3, 4, 6],
            [1, 5, 5, 1, 4],
            [5, 2, 6, 6, 4],
            [2, 4, 1, 1, 2]
        ]
    ),
    (6, 5): np.array(
        [
            [4, 2, 5, 1, 4, 3], 
            [3, 6, 3, 2, 3, 3], 
            [3, 4, 2, 5, 2, 1], 
            [2, 3, 5, 1, 2, 2], 
            [5, 2, 5, 4, 3, 3]
        ]
    )
}

def get_model_name(KWARGS, version=None, step=None) -> str:
    name_str = f'tos_{KWARGS["num_col"]}by{KWARGS["num_row"]}_{KWARGS["num_rune"]}color_{KWARGS["num_action"]}act_{KWARGS["max_move"]}move_{KWARGS["mode"]}_{KWARGS["extra_obs"]}obs'
    if version is not None:
        name_str += f'_{version}'
    if step is not None:
        name_str += f"_{step}step"

    return name_str

def get_ppo_name(KWARGS, version=None, step=None) -> str:
    return f'ppo_{get_model_name(KWARGS, version, step)}'

def get_ppo_path(KWARGS, step=None, version=None) -> str:
    if not os.path.exists(PPO_PATH):
        os.makedirs(PPO_PATH)
    return os.path.join(PPO_PATH, get_ppo_name(KWARGS, version, step))

def get_dqn_name(KWARGS, version=None, step=None, reward=None) -> tuple[str, str]:
    name_str = get_model_name(KWARGS, version, step)
    if reward is not None:
        name_str += f'_{reward}reward'
    return f'dqn_{name_str}_eval', f'dqn_{name_str}_target'

def get_dqn_path(KWARGS, dir=DQN_PATH, step=None, version=None, reward=None) -> tuple[str, str]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    eval_name, target_name = get_dqn_name(KWARGS, version, step, reward)
    return os.path.join(dir, eval_name), os.path.join(dir, target_name)

def get_kwargs_by_name(name) -> dict:
    
    args = name.split('_')
    num_col = int(args[2].split('by')[0])
    num_row = int(args[2].split('by')[1])
    num_rune = int(args[3].split('color')[0])
    num_action = int(args[4].split('act')[0])
    max_move = int(args[5].split('move')[0])
    mode = args[6]
    extra_obs = int(args[7].split('obs')[0])
    return {
        'num_col': num_col,
        'num_row': num_row,
        'num_rune': num_rune,
        'num_action': num_action,
        'max_move': max_move,
        'mode': mode,
        'extra_obs': extra_obs
    }

def MoveDir2str(dir: int) -> str:
    """convert MoveDir to string"""
    for d in MoveDir:
        if d.value == dir:
            return d.name
    raise 'invalid direction'

def int2MoveDir(dir: int) -> MoveDir:
    """convert int to MoveDir"""
    for d in MoveDir:
        if d.value == dir:
            return d
    raise 'invalid direction'

def int2MoveDir_str(dir: int) -> str:
    """convert int to MoveDir string"""
    for d in MoveDir:
        if d.value == dir:
            return d.name
    raise 'invalid direction'




def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f'{func.__name__} elapsed time: {elapsed}')
        return result
    return wrapper

def set_process_priority():
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)


# def print_board(self):
#     print('')
#     for row in self.board:
#         print('\033[0m|', end=' ')
#         for rune in row:
#             if rune == Runes.NONE.value:
#                 print('\033[0mX', end=' ')
#             else:
#                 print(f'\033[1m{Runes.int2color_code(rune)}●', end=' ')
#         print('\033[0m|')
#     print('')

def print_two_board(board: np.ndarray, board2: np.ndarray):
    if board.shape != board2.shape:
        raise ValueError('boards have different shapes')
    
    print('')
    for i in range(board.shape[0]):
        print('\033[0m|', end=' ')
        for j in range(board.shape[1]):
            if board[i, j] == Runes.NONE.value:
                print('\033[0mX', end=' ')
            else:
                print(f'\033[1m{Runes.int2color_code(board[i, j])}●', end=' ')
        print('\033[0m|', end='')
        if i == board.shape[0] // 2:
            print('  ->  ', end='')
        else:
            print('      ', end='')

        print('\033[0m|', end=' ')
        for j in range(board2.shape[1]):
            if board2[i, j] == Runes.NONE.value:
                print('\033[0mX', end=' ')
            else:
                print(f'\033[1m{Runes.int2color_code(board2[i, j])}●', end=' ')
        print('\033[0m|')

