from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent as ddpg_agent
from collections import deque
import torch
import matplotlib.pyplot as plt
import os
import pickle


def train():
    """
    Trains a DDPG agent in the Unity Reacher environment.
    """
    # set hyperparameters
    buffer_size = int(1e5)  # replay buffer size
    batch_size = 128        # minibatch size
    gamma = 0.99            # discount factor
    tau = 1e-3              # for soft update of target parameters
    lr_actor = 1e-4         # learning rate of the actor
    lr_critic = 1e-4        # learning rate of the critic
    weight_decay = 0        # L2 weight decay
    actor_fc1_units = 200   # actor network, size of fully connected layer 1
    actor_fc2_units = 100   # actor network, size of fully connected layer 2
    critic_fc1_units = 200  # critic network, size of fully connected layer 1
    critic_fc2_units = 100  # critic network, size of fully connected layer 2
    seed = 0

    # use a simple concatenation of all hyperparameters as the experiment name. results are stored in a subfolder
    #   with this name
    experiment_name = "production-exp-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-uhlenbeck_sigma0.2".format(
        buffer_size, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay, actor_fc1_units, actor_fc2_units,
        critic_fc1_units, critic_fc2_units, seed)

    # in addition to creating the experiment folder, create subfolders for checkpoints and logs
    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
        os.mkdir(experiment_name+'/checkpoints')
        os.mkdir(experiment_name+'/logs')

    # log the hyperparameters
    with open(experiment_name + '/logs/' + 'hyperparameters.log', 'w') as f:
        print("Buffer size {}\nbatch size {}\ngamma {}\ntau {}\nlr_actor {}\nlr_critic {}\nweight decay {}\nactor_fc1-fc2 {}-{}\ncritic_fc1-fc2 {}-{}\nseed {}".format(
            buffer_size, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay, actor_fc1_units, actor_fc2_units, critic_fc1_units, critic_fc2_units, seed), file=f)

    ############ THE ENVIRONMENT ###############
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get the number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # get the size of the action space
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent is:\n', states[0])

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


    # run the train loop
    scores_all = train_loop(env=env, brain_name=brain_name,
                            agent=ddpg, num_agents=num_agents,
                            experiment_name=experiment_name)

    pickle.dump(scores_all, open(experiment_name+'/scores_all.pkg', 'wb'))

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_all) + 1), scores_all)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # finally, close the environment
    env.close()


def train_loop(env, brain_name, agent, num_agents, experiment_name,
               n_episodes=1000, max_t=300, print_every=100):
    """
    Adopted from the Udacity pendulum DDPG implementation.
    """

    experiment_directory = experiment_name
    checkpoints_directory = experiment_directory + '/checkpoints/'
    log_directory = experiment_directory + '/logs/'

    logfile = open(log_directory+experiment_name+'.log', 'w')

    scores_deque = deque(maxlen=print_every)
    scores_all = []
    for i_episode in range(1, n_episodes + 1):
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        #for t in range(max_t):
        while True:
            actions = agent.act(states)  # select an action (for each agent)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            scores += rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        scores_deque.append(scores)
        scores_all.append(scores)

        print('Episode {} ... Reward: {:.3f} ... Average Reward: {:.3f}'.format(
            i_episode, np.mean(scores), np.mean(scores_deque)))
        print('Episode {} ... Reward: {:.3f} ... Average Reward: {:.3f}'.format(
            i_episode, np.mean(scores), np.mean(scores_deque)), file=logfile)

        # save every 10 episodes
        if i_episode % 25 == 0:
            print('saving checkpoint')
            torch.save(agent.actor_local.state_dict(),
                       checkpoints_directory+'checkpoint_actor-episode_{:04d}.pth'.format(i_episode))
            torch.save(agent.critic_local.state_dict(),
                       checkpoints_directory+'checkpoint_critic-episode_{:04d}.pth'.format(i_episode))

        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode - 100, np.mean(scores_deque)))
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode - 100, np.mean(scores_deque)), file=logfile)
            torch.save(agent.actor_local.state_dict(),
                       checkpoints_directory+'checkpoint_actor-final.pth')
            torch.save(agent.critic_local.state_dict(),
                       checkpoints_directory+'checkpoint_critic-final.pth')
            break


    logfile.close()

    return scores_all


if __name__ == '__main__':
    train()
