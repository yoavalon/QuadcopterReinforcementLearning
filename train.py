import numpy as np
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Discrete, Box

from ray.rllib.models.tf.misc import normc_initializer
import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.rllib.agents import ppo

import matplotlib.pyplot as plt
import os
tf = try_import_tf()

#from q4env import DroneEnv     #Render
from q4envQ import DroneEnv     #Background execution


class CustomModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        
        inter1 = tf.keras.layers.Dense(
            50,
            name="inter1",
            activation= None, #'tanh',
            kernel_initializer=normc_initializer(0.01))(self.inputs)

        #'''
        inter2 = tf.keras.layers.Dense(
            100,
            name="inter2",
            activation='tanh',
            kernel_initializer=normc_initializer(0.01))(inter1)

        inter3 = tf.keras.layers.Dense(
            50,
            name="inter3",
            activation='tanh',
            kernel_initializer=normc_initializer(0.01))(inter2)

        inter4 = tf.keras.layers.Dense(
            20,
            name="inter4",
            activation='tanh',
            kernel_initializer=normc_initializer(0.01))(inter3)

        
        CriticInter1 = tf.keras.layers.Dense(
            20,
            name="CriticInter1",
            activation= None, #'tanh',
            kernel_initializer=normc_initializer(0.01))(self.inputs)

        #ac = tf.keras.layers.Dropout(0.5)(inter5)
        #'''

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="actor",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(inter4)
        
        value_out = tf.keras.layers.Dense(
            1,
            name="critic",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(CriticInter1)
        
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ray.init(num_gpus=1)
ModelCatalog.register_custom_model("my_model", CustomModel)

trainer = ppo.PPOTrainer(
    env=DroneEnv, config={
    "model": {
            "custom_model": "my_model",
        },    
    "gamma": 0.95, 
    "lr" : 0.0001, 
    "num_workers": 0, 
})

while True:
    trainer.train()    

'''
p = './checkpoints'
for folder in os.listdir(p):
    for fname in os.listdir(os.path.join(p, folder)) :
        if fname.endswith('.tune_metadata'):            
            out = os.path.join(p, folder, fname)
            out = os.path.splitext(out)[0] #without extension
            trainer.restore(out)
            print('=======================================')
            print(out)
            print('checkpoint restored')
            break

it = 0
while True:
    trainer.train()
    it+=1

    if it % 15 == 0 : 
        checkpoint = trainer.save(p)
        print('===================================')
        print("checkpoint saved at: ", checkpoint)

'''