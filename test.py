from unityagents import UnityEnvironment
from ddpg_agent import Agent as ddpg_agent
import torch
import numpy as np



def test():
    # set hyperparameters
    buffer_size = int(1e5)  # replay buffer size
    batch_size = 128        # minibatch size
    gamma = 0.99            # discount factor
    tau = 1e-3              # for soft update of target parameters
    lr_actor = 1e-4         # learning rate of the actor
    lr_critic = 1e-4        # learning rate of the critic
    weight_decay = 0        # L2 weight decay
    actor_fc1_units = 256   # actor network, size of fully connected layer 1
    actor_fc2_units = 128   # actor network, size of fully connected layer 2
    critic_fc1_units = 256  # critic network, size of fully connected layer 1
    critic_fc2_units = 128  # critic network, size of fully connected layer 2
    seed = 0

    ############ THE ENVIRONMENT ###############
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get the number of agents
    num_agents = len(env_info.agents)

    # get the size of the action space
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    #random_agent(env=env, brain_name=brain_name, num_agents=num_agents, action_size=action_size)



    # initialize DDPG agent
    ddpg = ddpg_agent(state_size=state_size,
                      action_size=action_size,
                      num_agents=num_agents,
                      random_seed=seed,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      gamma=gamma,
                      tau=tau,
                      lr_actor=lr_actor,
                      lr_critic=lr_critic,
                      weight_decay=weight_decay,
                      actor_fc1_units=actor_fc1_units,
                      actor_fc2_units=actor_fc2_units,
                      critic_fc1_units=critic_fc1_units,
                      critic_fc2_units=critic_fc2_units
                      )


    ddpg.actor_local.load_state_dict(torch.load('checkpoint_actor-final.pth'))
    ddpg.critic_local.load_state_dict(torch.load('checkpoint_critic-final.pth'))

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    for i in range(200):
        actions = ddpg.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            break

if __name__ == '__main__':
    test()
