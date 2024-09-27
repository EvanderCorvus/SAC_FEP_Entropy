import numpy as np
from tqdm import tqdm

def train_episode(
        env, 
        agent, 
        replay_buffer, 
        current_episode, 
        config, 
        writer=None
    ):
    observation, _ = env.reset()
    step = 0
    #for step in range(config['n_steps']):
    while True:
        # Perform Actions
        action = agent.act(observation)
        # Environment Steps
        next_obs, reward, terminated, truncated, info = env.step(action)

        replay_buffer.add(
            observation,
            action,
            [reward],
            next_obs
        )
        observation = next_obs

        if step>=config['n_optim']:
            # Update Agents
            for _ in range(config['n_optim']):
                sample = replay_buffer.sample(agent.batch_size)
                loss_actor, loss_critic, entropy = agent.update(sample)
                if writer is not None:
                    writer.add_scalar(
                        'loss/actor',
                        loss_actor,
                        step + current_episode*200
                    )
                    writer.add_scalar(
                        'loss/critic',
                        loss_critic,
                        step + current_episode*200
                    )
                    writer.add_scalar(
                        'loss/entropy',
                        entropy,
                        step + current_episode*200
                    )
        step += 1

        if terminated or truncated:
            break
    return reward


def test_episode(
        env, 
        agents, 
        config
    ):
    observations = env.reset()
    list_states = [env.states]
    for _ in tqdm(range(config['test_steps']), leave=False):
        #actions = np.zeros((env.n_agents, 1), dtype=np.float64)
        actions = np.array([agents[f'agent_{i}'].act(observations[i]) for i in range(env.n_agents)])
        next_obs, _= env.active_brownian_rollout(actions, config['frame_skip'])
        observations = next_obs
        list_states.append(env.states)

    return list_states

