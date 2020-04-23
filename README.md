# Reinforcement Learning Quadcopter Environment 
Quadcopter-env

![Drone](/img/quadImg.png)

The Quadcopter-env is a **Physics Reinforcment Learning Environment** training a quadcopter to acccomplish various objectives. The environment was created using panda3d and it's Bullet a modern and open source physics engine (See https://www.panda3d.org/ and https://docs.panda3d.org/1.10/python/programming/physics/bullet/index)

Prerequesits : 

* Python 3.X
* panda3d `pip3 install Panda3D`
* Tensorflow 1.X
  
The environment simulates the physics of quadcopter movement. 
To solve the environment using Proximate Policy Optimization (PPO) (https://arxiv.org/abs/1707.06347) we use the Reinforcement Learning Framework RLlib. (https://ray.readthedocs.io/en/latest/rllib.html).

The Environment is highly customizeable and many other tasks can be implemented.

The environment q4envQ.py runs training in background without visualization and q4env.py has a visualization. Since models are stored using train.py we recommend using q4envQ.py for training, and to visualiza restore the model to q4env.py

To cite this repository in publications:

    @misc{Quadcopter-env,
      author = {Yoav Alon},
      title = {Quadcopter-env Reinforcement Learning Environment},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/yoavalon/Quadcopter-env}},
    }

## Approaching a target -task
In the first default task the environment is configured to motivate learning of moving to drone to a green target point.

![Drone](/img/droneTrace2.png)

When performing the training using PPO and a 3 dense layers together with a dropout layer the observation is that the initial policy learns to accelerate the drone into the right direction. In a further milestone the acceleration is addapted to land more close to the actual target. A much more advanced policy will accelerate the drone faster in the beginning, and counter accelerate it towards the goals. A higher episode length will motivate this behaviour more.

![Drone](/img/rews2.png)

## Multi-Drone Coordination
A single policy extracts actions that are split to multiple drones. Respective reward signals are returned. The same target point as in the previous task has to be achieved by multiple drones now. 
