import os
import time
import gymnasium as gym
from stable_baselines3 import DQN
from typing import Optional

from util import *

KWARGS_DQN = {
    'num_col': 6, 
    'num_row': 5, 
    'num_rune': 3, 
    'max_move': 1000, 
    'num_action': 9, 
    'mode': 'fixed',
    'extra_obs': 3
}

gym.register(
    id='TosEnv-ppo-v0',
    entry_point='tos_env:TosBaseEnv',
    max_episode_steps=1000,
    kwargs=KWARGS_DQN
)
print(f'register ppo env in train.py')


def main():

    set_process_priority()

    env = gym.make('TosEnv-ppo-v0')

    hidden_sizes = [512, 512]

    name = 'dqn_' + get_model_name(KWARGS_DQN, step=None, version='0.02Penal_noLimit_defaltkwarg')

    model = DQN(
        policy="MlpPolicy",
        env=env,
        batch_size=256,
        tensorboard_log=f"E:/dqn_logs/{name}_log",
        # policy_kwargs={
        #     "net_arch": hidden_sizes,
        # },
        verbose=1,
        # device='cuda'
    )
    total_step = 12800000
    save_step = 128000

    model_dir = f'E:/dqn_model/'
    os.makedirs(model_dir, exist_ok=True)

    def save_name(name, step: int) -> str:
        return f'{model_dir}{name}_{step}step.zip'
    
    # log_dir = f'E:/ppo_logs/{name}_log'
    # os.makedirs(log_dir, exist_ok=True)

    def callback(locals, globals):
        if locals['self'].num_timesteps % (save_step) == 0:
            locals['self'].save(save_name(name, locals['self'].num_timesteps))
            print(f'save model at {locals["self"].num_timesteps} step')
        return True
    
    # if os.path.exists(f"{name}"):
    #     model.load(f"{name}")
    #     print('Model found, continue training...')
    # else:
    #     print('Model not found, start new training...')

    model.learn(total_timesteps=total_step, progress_bar=True, callback=callback, log_interval=500)
    # model.save(name)

if __name__ == '__main__':
    main()
        

