import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Optional

from utils import get_ppo_name, set_process_priority

KWARGS = {
    'num_col': 6, 
    'num_row': 5, 
    'num_rune': 3, 
    'max_move': 1000, 
    'num_action': 9, 
    'mode': 'fixed',
    'extra_obs': 3
}
# KWARGS = {
#     'num_col': 3, 
#     'num_row': 3, 
#     'num_rune': 3, 
#     'max_move': 1000, 
#     'num_action': 9, 
#     'mode': 'fixed',
#     'extra_obs': 3
# }

gym.register(
    id='TosEnv-ppo-v0',
    entry_point='tos_env:TosBaseEnv',
    max_episode_steps=1000,
    kwargs=KWARGS
)
print(f'register ppo env in train.py')


def main():

    set_process_priority()

    env = gym.make('TosEnv-ppo-v0')

    hidden_sizes = [512, 512]
    lr = 3e-4
    n_envs = 1
    step_per_update = 3000
    batch_size = n_envs * step_per_update # 256
    repeat_per_update = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.1 # 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    name = get_ppo_name(KWARGS, step=None, version='0.02Penal_noLimit_defaltkwarg')

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={
            "net_arch": {
                "pi": hidden_sizes,
                "vf": hidden_sizes,
            },
        },
        learning_rate=lr,
        n_steps=step_per_update * n_envs,
        batch_size=batch_size,
        n_epochs=repeat_per_update,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=True,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device='cpu',
        tensorboard_log=f"E:/ppo_logs/{name}_log",
    )

    total_step = 12800000
    save_step = 128000
    

    model_dir = f'E:/ppo_model/'
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

    model.learn(total_timesteps=total_step, progress_bar=True, callback=callback, log_interval=5)
    # model.save(name)

if __name__ == '__main__':
    main()
        

