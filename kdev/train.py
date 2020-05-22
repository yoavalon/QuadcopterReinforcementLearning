import tensorflow as tf
import numpy as np
from env4 import DroneEnv
from ppo import PPO
from buffer import Buffer
import configs

env = DroneEnv()
ppo = PPO()
buffer = Buffer()

while(True) :
    s = env.reset()

    while(True):

        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)

        buffer.append(s,a,r)
        s = s_

        if (env.t+1) % configs.BATCH == 0 :

            v = ppo.getValue(s_)
            buffer.discount(v)
            #print(f'{v}   {np.linalg.norm(s[0:3]-s[3:6])}   {s}   ')

            f = buffer.format()
            ppo.update(f[0], f[1], f[2])
            buffer.clear()

        if done :
            break
