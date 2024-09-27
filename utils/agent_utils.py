import torch as tr
import torch.nn as nn
import numpy as np
import random
import operator
from tensordict import TensorDict
import torch as tr
from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage, LazyMemmapStorage

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
# Source: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

class CustomReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state):
        data = (state, action, reward, next_state)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, action, reward, next_state = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def sample(self, batch_size):
        storage_length = len(self._storage)
        idxes = [random.randint(0, storage_length - 1) for _ in range(min(storage_length, batch_size))]
        return self._encode_sample(idxes)




class MultiAgentReplayBuffer(ReplayBuffer):
    def __init__(self, size, batch_size, batch_dim):
        self.batch_dim = batch_dim
        super().__init__(storage=LazyTensorStorage(size),
                         batch_size=batch_size,
                         )
    def add_transition(self, obs, act, reward, next_obs):
        data = self._create_td_transition(obs, act, reward, next_obs)
        self.add(data)

    # ToDo: Check if this can be done with torch.view or other reshaping methods
    def sample_agent_batches(self):
        data = self.sample().float().to(device)
        transitions = []
        for i in range(data['observation'].shape[1]):
            observation = data['observation'][i]
            action = data['action'][i]
            reward = data['reward'][i]
            next_observation = data['next_observation'][i]
            #done = data['done'][:, i, :]
            transitions.append((observation, action, reward, next_observation))

        return transitions

    def _create_td_transition(self, obs, act, reward, next_obs):
        transition = TensorDict({
            'observation': obs,
            'action': act,
            'reward': reward,
            'next_observation': next_obs
        },
            batch_size=self.batch_dim, # Here Batch dim is n_agents
            #device=device,
        )
        return transition
