import gymnasium as gym
from utils.utils import *
from utils.agent_utils import CustomReplayBuffer
from utils.train_utils import train_episode, test_episode
from models import SoftActorCritic
import torch as tr
from torch.utils.tensorboard.writer import SummaryWriter
import os
from tqdm import tqdm
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')


agent_config = hyperparams_dict('Agent', './hyperparameters.ini')
train_config = hyperparams_dict('Training', './hyperparameters.ini')
env = gym.make('Pendulum-v1')
agent = SoftActorCritic(agent_config, device).to(device)
replay_buffer = CustomReplayBuffer(agent_config['buffer_size'])

experiment_number = len(os.listdir('experiments'))+1
writer = SummaryWriter(f'experiments/{experiment_number}')

for episode in tqdm(range(train_config['n_episodes'])):
    reward = train_episode(env, agent, replay_buffer, episode, train_config, writer)
    writer.add_scalar('reward', reward, episode)

writer.flush()
writer.close()
env.close()