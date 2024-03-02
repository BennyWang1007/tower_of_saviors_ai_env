from Board import *
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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

BATCH_SIZE = 128
LR = 0.01
EPSILON = 0.9
GAMMA = 0.99
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

N_ACTIONS = KWARGS_DQN['num_action']
N_STATES = KWARGS_DQN['num_col'] * KWARGS_DQN['num_row'] + KWARGS_DQN['extra_obs']

gym.register(
    id='TosEnv-dqn-v0',
    entry_point='tos_env:TosBaseEnv',
    max_episode_steps=1000,
    kwargs=KWARGS_DQN
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action
    
    def predict(self, x, deterministic=True):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, eval_net_path='eval_net.pth', target_net_path='target_net.pth'):
        torch.save(self.eval_net.state_dict(), eval_net_path)
        torch.save(self.target_net.state_dict(), target_net_path)

    def load(self, eval_net_path='eval_net.pth', target_net_path='target_net.pth'):
        self.eval_net.load_state_dict(torch.load(eval_net_path))
        self.target_net.load_state_dict(torch.load(target_net_path))

    @staticmethod
    def load2(path=''):
        dqn = DQN()
        dqn.load(f'{path}_eval.pth', f'{path}_target.pth')
        return dqn



if __name__ == '__main__':

    env = gym.make('TosEnv-dqn-v0').unwrapped

    dqn = DQN()
    episode_number = 100000 * 10
    print_devider = 2000
    save_devider = 10000
    rewards = []

    for i in tqdm(range(episode_number)):
        
        s, info = env.reset()
        episode_reward_sum = 0.0

        while True:
            # env.render()
            a = dqn.choose_action(s)
            s_, r, terminal, _, info = env.step(a)

            dqn.store_transition(s, a, r, s_)
            episode_reward_sum += float(r)
            rewards.append(episode_reward_sum)

            s = s_

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if terminal:
                break

        if i % print_devider == 0 and i > 0:
            mean_reward = round(np.mean(rewards), 2)
            print(f'episode{i} -> mean_reward: {mean_reward}')
            rewards = []

        if i % save_devider == 0 and i > 0:
            model_dir = 'dqn_model'
            eval_name, target_name = get_dqn_path(KWARGS_DQN, dir=model_dir, step=i)
            dqn.save(eval_name, target_name)
    eval_name, target_name = get_dqn_name(KWARGS_DQN)
    dqn.save(eval_name, target_name)


