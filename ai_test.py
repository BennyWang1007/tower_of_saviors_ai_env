from Board import *

import numpy as np
from stable_baselines3 import PPO, DQN
import train_dqn
import gymnasium as gym

from animation import BoardScreen

from train import KWARGS
from util import *


def get_pos_by_obs(obs: np.ndarray, num_col: int, num_row: int) -> tuple[int, int]:
    return obs[num_col * num_row], obs[num_col * num_row + 1]

@timeit
def test_model_score(model: PPO | DQN | train_dqn.DQN, KWARGS: dict, env: gym.Env) -> tuple[tuple[float, int], tuple[float, int]]:

    # board = Board(**KWARGS)
    # bs = BoardScreen(board)
    
    num_col = KWARGS['num_col']
    num_row = KWARGS['num_row']

    def test_game(start_x, start_y) -> float:
        
        rewards: list = []
        env.reset()
        obs, info = env.reset_for_test(start_x, start_y)
        terminal: bool = False

        while not terminal:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminal, _, info = env.step(action)
            rewards.append(round(reward, 2))
        
        return np.sum(rewards)
    
    scores = []
    scores_outliners = []

    for x in range(num_col):
        for y in range(num_row):
            score = test_game(x, y)
            if score < 0:
                scores_outliners.append(score)
            else:
                scores.append(score)

    if len(scores) == 0:
        mean_score = 0.0
    else:
        mean_score = np.mean(scores)
    if len(scores_outliners) == 0:
        mean_score_outliners = 0.0
    else:
        mean_score_outliners = np.mean(scores_outliners)
    mean_score = round(mean_score, 2)
    mean_score_outliners = round(mean_score_outliners, 2)
    # print(f'm, m_o = {mean_score=}, {mean_score_outliners=}')
    print(f'{mean_score=} with {len(scores)} scores')
    print(f'{mean_score_outliners=} with {len(scores_outliners)} outliners')
    return (mean_score, len(scores)), (mean_score_outliners, len(scores_outliners))

if __name__ == "__main__":

    model_type = 'dqn'

    env = gym.make('TosEnv-ppo-v0')

    model_dir = 'E:\dqn_model2'
    step: int = 0
    model_scores: dict[str, tuple[float, float]] = {}

    for model_name in os.listdir(model_dir):
        if not model_name.endswith('.zip'):
            continue
        if 'defaltkwarg' in model_name:
            continue
        step = int(model_name.split('_')[-1].split('step')[0])
        # if step != 10112000:
        #     continue
        if model_type == 'dqn':
            model = DQN.load(f'{model_dir}/{model_name}')
        else:
            model = PPO.load(f'{model_dir}/{model_name}')
        model.device = 'cuda'
        KWARGS = get_kwargs_by_name(model_name)
        # print(f'load model {model_name}')
        # print(f'{KWARGS=}')
        model_scores[step] = test_model_score(model, KWARGS, env)
        # break

    print(f'{model_scores=}')