import itertools
import numpy as np
import matplotlib
import sys
import tensorflow as tf
import collections

from Actor import Actor
from Critic import Critic
from q4env import DroneEnv

env = DroneEnv()

def multi_critic(env, actor, discount_factor=0.95):

    CriticLoss1 = 0
    CriticLoss2 = 0
    ActorLoss = 0

    for i_episode in range(100000):

        state = env.reset()

        ep_r = 0
        ep_r1 = 0
        ep_r2 = 0

        for step in itertools.count():

            action = actor.predict(state)

            next_state, reward1, reward2, done, _ = env.step(action)

            if done :
                break

            ep_r += reward1 + reward2
            ep_r1 += reward1
            ep_r2 += reward2

            #Critic1
            value_next = critic1.predict(next_state)
            td_target = reward1 + discount_factor * value_next
            td_error1 = td_target - critic1.predict(state)

            td_target = np.expand_dims(td_target,0) #ADDED
            td_target = np.expand_dims(td_target,0) #ADDED

            CriticLoss1 = critic1.update(state, td_target)

            #Critic2
            value_next2 = critic2.predict(next_state)
            td_target2 = reward2 + discount_factor * value_next2
            td_error2 = td_target2 - critic2.predict(state)

            td_target2 = np.expand_dims(td_target2,0) #ADDED
            td_target2 = np.expand_dims(td_target2,0) #ADDED

            CriticLoss2 = critic2.update(state, td_target2)

            #KEY POINT ==================================
            errors = [td_error1,td_error2]
            td_error = np.min(errors)
            #============================================

            td_error = np.expand_dims(td_error,0) #ADDED
            td_error = np.expand_dims(td_error,0) #ADDED

            action = np.expand_dims(action, 0)

            #td error here should be called advantage
            ActorLoss = actor.update(state, td_error, action)

            state = next_state

        summary = tf.Summary(value=[tf.Summary.Value(tag='first/reward', simple_value=ep_r)])
        summary1 = tf.Summary(value=[tf.Summary.Value(tag='r/r1', simple_value=ep_r1)])
        summary2 = tf.Summary(value=[tf.Summary.Value(tag='r/r2', simple_value=ep_r2)])
        summary3 = tf.Summary(value=[tf.Summary.Value(tag='first/steps', simple_value=step)])
        summary4 = tf.Summary(value=[tf.Summary.Value(tag='loss/critic1Loss', simple_value=CriticLoss1)])
        summary5 = tf.Summary(value=[tf.Summary.Value(tag='loss/critic2Loss', simple_value=CriticLoss2)])
        summary6 = tf.Summary(value=[tf.Summary.Value(tag='loss/ActorLoss', simple_value=np.sum(np.abs(ActorLoss)))])

        writer.add_summary(summary3, i_episode)
        writer.add_summary(summary, i_episode)
        writer.add_summary(summary1, i_episode)
        writer.add_summary(summary2, i_episode)
        writer.add_summary(summary3, i_episode)
        writer.add_summary(summary4, i_episode)
        writer.add_summary(summary5, i_episode)
        writer.add_summary(summary6, i_episode)


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
actor = Actor()

critic1 = Critic(scope = 'Critic_1')
critic2 = Critic(scope = 'Critic_2')

writer = tf.summary.FileWriter('log', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    multi_critic(env, actor)
