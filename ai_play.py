from Board import *

import numpy as np
from stable_baselines3 import PPO, DQN
import stable_baselines3
import gymnasium as gym

from animation import BoardScreen

from train import KWARGS
from train_dqn import KWARGS_DQN
import train_dqn
from util import *


# gym.register(
#     id='TosEnv-ppo-v0',
#     entry_point='tos_env:TosBaseEnv',
#     max_episode_steps=1000, 
#     kwargs=KWARGS
# )
# print(f'register ppo env in ai_play.py')

# gym.register(
#     id='TosEnv-dqn-v0',
#     entry_point='tos_env:TosBaseEnv',
#     max_episode_steps=1000,
#     kwargs=KWARGS_DQN
# )
# print(f'register dqn env in ai_play.py')

num_col = KWARGS['num_col']
num_row = KWARGS['num_row']
num_rune = KWARGS['num_rune']
max_move = KWARGS['max_move']
num_action = KWARGS['num_action']
mode = KWARGS['mode']
extra_obs = KWARGS['extra_obs']


def test_model(model: PPO | DQN | train_dqn.DQN, KWARGS: dict) -> None:

    if isinstance(model, PPO):
        env = gym.make('TosEnv-ppo-v0')
    elif isinstance(model, DQN):
        env = gym.make('TosEnv-ppo-v0')
    elif isinstance(model, train_dqn.DQN):
        env = gym.make('TosEnv-ppo-v0')
    else:
        raise ValueError(f'Invalid model type: {type(model)}')

    board = Board(**KWARGS)
    bs = BoardScreen(board)

    def test_game() -> None:
        
        actions: list[int] = []
        rewards: list = []
        obs, info = env.reset()
        terminal: bool = False
        board.set_by_obs(obs)

        while not terminal:
            if isinstance(model, train_dqn.DQN):
                action = model.predict(obs)
            else:
                action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminal, _, info = env.step(action)
            actions.append(action.item())
            rewards.append(round(reward, 2))

        action_str = "".join([MoveDir.int2str(action).ljust(3) + ', ' for action in actions])
        print(f'actions=[{action_str[:-2]}]')
        print(f'rewards={rewards}')
        print(f'acc_reward={np.sum(rewards)}')
        bs.set_board(board)
        bs.set_actions(actions)
        bs.start_move()
    
    bs.set_next_callback(test_game)
    test_game()


def test_record() -> None:

    env = gym.make('TosEnv-ppo-v0')

    pre_record_str = 'L L RU'.split(' ')
    pre_record = [MoveDir.str2int(action) for action in pre_record_str]

    def test_game() -> None:

        record_idx = 0
        actions: list[int] = []
        rewards: list = []

        obs, info = env.reset()
        terminal: bool = False
        board = get_board(obs, num_col, num_row, num_rune, max_move, num_action)

        while not terminal and record_idx < len(pre_record):
            action = pre_record[record_idx]
            obs, reward, terminal, _, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            record_idx += 1
    
        print(f'actions={actions}')
        print(f'rewards={rewards}')
        bs.set_board(board)
        bs.set_actions(actions)
        bs.start_move()

    bs = BoardScreen(Board(num_col, num_row, num_rune, max_move, num_action, mode))
    bs.set_next_callback(test_game)
    test_game()
    
if __name__ == "__main__":
    model_dir = 'ppo_model'
    model_name = 'ppo_tos_6by5_3color_9act_1000move_fixed_3obs_0.02Penal_noLimit'
    model = PPO.load(os.path.join(model_dir, model_name))

    # model_dir = 'dqn_model'
    # model_name = 'dqn_tos_6by5_3color_9act_1000move_fixed_3obs_0.02Penal_noLimit_defaltkwarg'
    # model = DQN.load(os.path.join(model_dir, model_name))

    # model_dir = 'dqn_model'
    # model_name = 'dqn_tos_6by5_3color_9act_1000move_fixed_3obs'
    # model = train_dqn.DQN.load2(os.path.join(model_dir, model_name))
    
    KWARGS = get_kwargs_by_name(model_name)

    gym.register(
        id='TosEnv-ppo-v0',
        entry_point='tos_env:TosBaseEnv',
        max_episode_steps=1000, 
        kwargs=KWARGS
    )
    test_model(model, KWARGS)
    # test_record()