"""
Runs a trained DQN in an environment and plots results at /date/exp_name.
"""
import torch
import torch.nn as nn
import gym
import os
import utils.nnetwork as dqn
from utils.atari_wrapper import make_atari, wrap_deepmind
from utils.evaluate import act_for_one_episode,plot,plot_hist
import matplotlib.pyplot as plt
import numpy as np

# Main: Script for evaluating a trained DQN performance
env_name = 'BreakoutNoFrameskip-v4'
cwd = os.getcwd()

# Date specifies the training date of the network to be evaluated
date = '08_13_23_27' #Scheme 2
exp_name = f'{date}_{env_name}'
LTL_game= False
LTL_net = False
model_path = f'/data/{exp_name}/pyt_save/model.pt'
print(f"Running the model found at {model_path}, which has LTL_Net = {LTL_net}, and LTL_Game={LTL_game}")
plt_path = f'/data/{exp_name}/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 1000
env = make_atari(env_name)
env = wrap_deepmind(env, episode_life=False, clip_rewards = False, frame_stack=True, scale=True)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
# get the saved neural net
if LTL_net:
    policy_net = dqn.FCombined_NN(obs_shape[:2],n_actions,n_qstates=4).to(device)
else:
    policy_net = dqn.CNN(obs_shape[:2],n_actions).to(device)
policy_net.load_state_dict(torch.load(cwd+model_path,map_location=device))

follow_rets,final_q_states = [],[]

for i in range(episodes):
    ep_ret,final_percent_blue = act_for_one_episode(net=policy_net,environment=env, 
                                        epsilon = 5,
                                        render=False,
                                        LTL=LTL_net,
                                        LTL_Game = LTL_game,
                                        starting_q_state=[1,0,0,0])
    follow_rets.append(ep_ret)
    final_q_states.append(np.round(final_percent_blue*100))
fig,qfig = plt.figure(),plt.figure()
plot(fig,follow_rets,'Returns',cwd+plt_path)
plot(qfig,final_percent_blue,'Final Percent Blue',cwd+plt_path)
print(f'average reward is {np.mean(follow_rets)}')
print(f'average percent blue remaining is {np.mean(final_q_states)}')
