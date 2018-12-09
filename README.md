<img align="center" src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif">

# Reacher - Training a Double-Jointed Arm with Deep RL

## Environment Details
In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of an agent is to maintain its position at the target location for as many
time steps as possible.

The environment contains 20 identical agents, each with its own copy of the environment.

### State Space
The observation space consists of 33 variables corresponding to position, rotation, velocity,
and angular velocities of the arm. 

### Action Space
Each action is a vector with four numbers, corresponding 
to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solution Criterium
The agents must get an average score of +30 over 100 consecutive episodes, and over all agents.

- After each episode, the rewards that each agent received are added up (without discounting),
 to get a score for each agent. 
 This yields 20 (potentially different) scores. The average is taken of these 20 scores.
 - This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of 
those average scores is at least +30. In the case of the plot above, 
the environment was solved at episode 63, since the average of the average scores 
from episodes 64 to 163 (inclusive) was greater than +30.

## How To Run
### Dependencies

1. Install the dependencies as specified on the Udacity DRLND Github repository [here](https://github.com/udacity/deep-reinforcement-learning)

2. Download the environment from one of the links below. You need only select
   the environment that matches your operating system:

    - **_Version: Twenty (20) Agents_**
        - Linux: [click
          here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click
          here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click
          here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click
          here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
3. Place the file in the working folder, and unzip the file.

4. Clone this repository.


### How To Run The Code
Run `python train.py` to start training or `python test.py` to observe how the trained agent performs with 
the checkpoints included.

