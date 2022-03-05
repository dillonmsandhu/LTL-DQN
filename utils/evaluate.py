import torch
import torch.nn as nn
import gym
from . import nnetwork as dqn
from .atari_wrapper import make_atari, wrap_deepmind
import utils.alternative_ltl as ltl 
import numpy as np
import matplotlib.pyplot as plt
import time

def percent_blue(s,p=.5):
    """At least p percent of initially blue pixels are blue"""
    blue_pixels = np.sum(s[35:37,5:-5] == 1/3)
    return blue_pixels / 148

def act_for_one_episode(net,environment,epsilon=5,render=False,LTL=False,starting_q_state=None,LTL_Game=False):
    ep_rews = []          # for measuring episode returns
    # Start episode.    
    s = environment.reset()      # first obs comes from starting distribution
    prior_q_state = starting_q_state # initial state of the NBDA
    finished = False             # signal from environment that episode is over
    while not finished:
        if not LTL:
            a = dqn.get_action(dqn.to_tensor([s]),epsilon,net,environment)
        if LTL:
            a = dqn.get_action(dqn.to_tensor([s]),epsilon,net,environment,q_state = dqn.to_tensor([prior_q_state])) 
        s_next, reward, finished, _ = environment.step(a)
        q_state = ltl.dynamics(prior_q_state, ltl.Label(s.frame(3)))
        next_q_state = ltl.dynamics(q_state, ltl.Label(s_next.frame(3)))
        
        if LTL_Game: # play the LTL Version of the game
            # end episode if LTL criteria is not met.
            if next_q_state == [0,0,1,0] or next_q_state == [0,0,0,1]:
                finished = True
                if next_q_state == [0,0,0,1]:
                    reward = 10000
                if next_q_state == [0,0,1,0]:
                    reward = 0
        
        ep_rews.append(reward)
        if render:
            environment.render()
            time.sleep(.012)    
        # move to the next transition
        s = s_next
        prior_q_state = next_q_state

    final_percent_blue = percent_blue(s_next.frame(3))
    print(final_percent_blue)
    return sum(ep_rews),final_percent_blue

def plot(fig,y,plt_name,plt_dir):
    plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(y)
    fig.savefig(plt_dir + f'{plt_name}.png')

def plot_hist(fig,y,plt_name,plt_dir):
    plt.close(fig)
    fig = plt.figure()
    plt.hist(y, density=True, bins=len(y))  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Final Q State')
    fig.savefig(plt_dir + f'{plt_name}.png')