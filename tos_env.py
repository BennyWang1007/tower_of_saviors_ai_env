from Board import *

from abc import ABC
from typing import Optional, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TosBaseEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, num_col: int, num_row: int, num_rune: int, max_move: int, num_action: int, mode: Literal['fixed', 'random'], extra_obs: int) -> None:
        print(f'init TosBaseEnv')
        print(f'{num_col=}, {num_row=}, {num_rune=}, {max_move=}, {num_action=}')
        self.num_col = num_col
        self.num_row = num_row
        self.num_rune = num_rune
        self.max_move = max_move
        self.num_action = num_action
        self.mode = mode

        self.board = Board(num_col, num_row, num_rune, max_move, num_action, mode, extra_obs)
        self.board.random_board()
        self.board.random_pos()
        self.action_space = spaces.Discrete(num_action)
        # self.observation_space = spaces.Box(low=0, high=num_col*num_row, shape=(num_col*num_row+extra_obs,), dtype=np.int8)
        if extra_obs > 3:
            self.observation_space = spaces.Box(low=0, high=num_col*num_row, shape=(num_col*num_row+extra_obs,), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=max(num_col, num_row, num_action, num_rune+1), shape=(num_col*num_row+extra_obs,), dtype=np.int8)
        # state_vec = [num_rune] * (num_col * num_row) + [num_col] + [num_row] + [num_action]
        # if extra_obs > 3:
        #     state_vec += [max_move]
        # self.observation_space = spaces.MultiDiscrete(state_vec)
        # self.observation_space = spaces.MultiDiscrete(state_vec)
        self.board.reset()
        self.reward = 0.0
        self.obs = self.get_obs()
        self.prev_reward = 0.0
        self.action = 0

    def set_pos(self, x: int, y: int) -> None:
        self.board.set_cur_pos(x, y)

    def get_obs(self) -> np.ndarray:
        return self.board.get_obs()

    def render(self) -> None:
        print(f'{self.num_col=}, {self.num_row=}, {self.num_rune=}, {self.max_move=}, {self.num_action=}')
        self.board.print_board()

    def close(self) -> None:
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        self.board.reset()
        self.board.random_pos()
        self.obs = self.get_obs()
        self.reward = 0.0
        self.prev_reward = 0.0
        
        return self.obs, {}
    
    def reset_for_test(self, x: int, y: int) -> tuple[np.ndarray, dict]:
        self.board.reset()
        self.board.set_cur_pos(x, y)
        self.obs = self.get_obs()
        self.reward = 0.0
        self.prev_reward = 0.0
        
        return self.obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.prev_reward = self.reward
        self.action = action
        self.board.move(action)
        self.board.evaluate()
        self.obs = self.get_obs()
        terminal = self.board.is_game_over()
        
        # calculate reward
        board_reward = self.board.first_combo * 0.8 + self.board.combo * 2.0 + self.board.totol_eliminated * 0.4 # + self.board.move_count * 0.05
        move_panalty = 0.02
        # print(f'board_reward={board_reward}, prev_reward={self.prev_reward}')
        # print(f'act/pre_act={action}/{self.board.prev_action}, oppo={MoveDir.opposite(self.board.prev_action)}')
        reward = board_reward - self.prev_reward
        if self.board.action_invalid:
            reward -= 5.0
        if self.action == MoveDir.opposite(self.board.prev_action):
            reward -= 3.0
        elif action == MoveDir.NONE.value or terminal:
            pass
        else:
            reward -= move_panalty

        self.reward = board_reward
        
        return self.obs, reward, terminal, False, {}


