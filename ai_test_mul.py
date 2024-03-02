
import concurrent.futures
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
import train_dqn
import random
import os


MAX_PROCCESSES = 7

def test_model_score(model_paths: list, type: str, KWARGS: dict, env: gym.Env) -> dict:

    results: dict = {}

    for name in model_paths:

        model: PPO | DQN | train_dqn.DQN
        if type == 'ppo':
            model = PPO.load(name)
        elif type == 'dqn':
            model = DQN.load(name)
        elif type == 'dqn_train':
            model = train_dqn.DQN.load2(name)
        else:
            raise ValueError('invalid model type')
        if isinstance(model, PPO) or isinstance(model, DQN):
            model.device = 'cpu'
        step = name.split('_')[-1].split('step')[0]
        print(f'loaded {step}')
        
        def test_game(start_x, start_y) -> float:
            
            rewards: list = []
            env.reset()
            obs, info = env.reset_for_test(start_x, start_y)
            terminal: bool = False

            while not terminal:
                if type == 'dqn_train':
                    action = model.predict(obs)
                else:
                    action = model.predict(obs, deterministic=True)[0]
                obs, reward, terminal, _, info = env.step(action)
                rewards.append(round(reward, 2))
            
            return np.sum(rewards)
        
        scores = []
        scores_outliners = []

        for x in range(KWARGS['num_col']):
            for y in range(KWARGS['num_row']):
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

        # print(f'{mean_score=} with {len(scores)} scores')
        # print(f'{mean_score_outliners=} with {len(scores_outliners)} scores')
        
        results[step] = (mean_score, len(scores)), (mean_score_outliners, len(scores_outliners))
        print(f'{step}: {results[step]}')

    return results

    

if __name__ == "__main__":

    model_type = 'dqn_train'
    model_dir = 'dqn_model'
    model_names = []
    model_paths = []
    sep_model_paths = []
    args: list[tuple] = []

    for model_name in os.listdir(model_dir):

        # if not model_name.endswith('.zip'):
        if not model_name.endswith('eval.pth'):
            continue

        # remove .zip
        # model_name = model_name[:-4]

        # remove _eval.pth
        model_name = model_name[:-9]

        model_path = os.path.join(model_dir, model_name)

        model_names.append(model_name)
        model_paths.append(model_path)

    KWARGS = train_dqn.get_kwargs_by_name(model_names[0])
    KWARGS['max_move'] = 20
    gym.register(
        id='TosEnv-ppo-v0',
        entry_point='tos_env:TosBaseEnv',
        max_episode_steps=1000,
        kwargs=KWARGS
    )

    # separate model names into MAX_PROCCESSES lists
    for i in range(MAX_PROCCESSES):
        sep_model_paths.append(model_paths[i::MAX_PROCCESSES])

    for model_paths in sep_model_paths:
        args.append((model_paths, model_type, KWARGS, gym.make('TosEnv-ppo-v0')))
        # print(f'{len(model_names)=}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCCESSES) as executor:
        results = executor.map(test_model_score, *zip(*args))
        combined_results = {}
        for result in results:
            for key, value in result.items():
                key = int(key)
                combined_results[key] = value
        sorted_results = dict(sorted(combined_results.items(), key=lambda item: item[0]))

        print(sorted_results)
        print('done')





