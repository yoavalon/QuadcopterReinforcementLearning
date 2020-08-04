# Reinforcement Learning Quadcopter Environment
Quadcopter-env

![Drone](/img/Selection_077.png)

The Quadcopter-env is a **Physics Reinforcement Learning Environment** training a quadcopter to acomplish various objectives. The environment was created using panda3d and it's Bullet a modern and open source physics engine (See https://www.panda3d.org/ and https://docs.panda3d.org/1.10/python/programming/physics/bullet/index)

Prerequesits :

* Python 3.X
* panda3d `pip3 install Panda3D`
* Tensorflow (or other prefered Deep Learning Framework)

The environment simulates the physics of quadcopter movement.
To solve the environment using Proximate Policy Optimization (PPO) (https://arxiv.org/abs/1707.06347) we use the Reinforcement Learning Framework RLlib. (https://ray.readthedocs.io/en/latest/rllib.html).

![Drone](/img/quadImg.png)

The Environment is highly customizeable and many other tasks can be implemented.

The environment can be run without rendering for training and for visualization with rendering.

In order to render the environment in q4env.py simply set:

    self.visualize = True

For no rendering and quick trainig in background set:

    self.visualize = False

To restore a previous checkpoint simpy uncomment restore() in train.py. The train() function will not create checkpoints where as the trainCheckpoint() function will.

    #restore()
    trainCheckpoint()
    #train()

To cite this repository in publications:

    @misc{Quadcopter-env,
      author = {Yoav Alon},
      title = {Quadcopter-env Reinforcement Learning Environment},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/yoavalon/Quadcopter-env}},
    }

## Approaching a target -task (/taskNavigate)
In the first default task the environment is configured to motivate learning of moving to drone to a green target point.

![Drone](/img/singleQuad.png)

When performing the training using PPO and a 3 dense layers together with a dropout layer the observation is that the initial policy learns to accelerate the drone into the right direction. In a further milestone the acceleration is addapted to land more close to the actual target. A much more advanced policy will accelerate the drone faster in the beginning, and counter accelerate it towards the goals. A higher episode length will motivate this behaviour more.

The observation- and action space are defined as :
        self.action_space = Box(-1,1,shape=(3,))  
        self.observation_space = Box(-50,50,shape=(6,))

For 3 continous actions that control the drones inclination force. The 6 observation dimensions return 3 dimensional position and 3 dimensional velocity. The environment is easily adaptable to different forms of control such as engine power in a 4 dimensional array, then the force has to be calculated.

![Drone](/img/rews80.png)
Mean reward signal over episode.

## Balance the Quadcopter (/taskBalance)

In this task the action space has 4 dimensions and allows the enginge control. With constant reward signal the episode length becomes the decisive criteria and the target is to balance the drone as long as possible. By adding randim impulses wind can be simulated.

## Multi-Agent Coordination (/taskMulti)
This task is motivated my Multi-Agent Reinforcement Learning environments that extract multiple reward signals and a single policy extracts actions that are split to multiple drones. Respective multiple reward signals are returned. The same target point as in the previous task has to be achieved by multiple drones now.
The challenge here is to construct networks that are able to perform a distinctive value estimation. A standard actor-critic reinforcement learning appraoch would fail here and optimize parts of the policy that have a dominant reward extraction.

![Drone](/img/quadImg.png)
