import os
from datetime import datetime
from collections import namedtuple
import numpy as np
import torch
from torch.optim import Adam
import utils.nnetwork as dqn
from utils.nnetwork import get_q, device, to_tensor
from utils.replay_memory import ReplayMemory
from utils.atari_wrapper import make_atari,wrap_deepmind
from utils import logger
from utils.evaluate import act_for_one_episode,plot
import matplotlib.pyplot as plt # for plotting durnig training
import utils.alternative_ltl as ltl

# Set up Environment
env_name = 'BreakoutNoFrameskip-v4' # pick atari game here.
exp_name = datetime.now().strftime("%m_%d_%-H_%-M_") + env_name
env = make_atari(env_name)
env = wrap_deepmind(env,episode_life=False, clip_rewards = False,frame_stack=True, scale=True) # parses the environment as in the 2014 paper
n_actions = env.action_space.n
obs_shape = env.observation_space.shape
cwd = os.getcwd() # For plotting

def train(num_frames, batch_size, gamma, replay_memory_size, replay_start_size,target_update_frequency, ep_start, 
            ep_end,eps_decay,train_freq=1,logger_kwargs=dict(),starting_model=None,model_save_frequency=None, 
            n_episodes_plot=None,act_epsilon=None,nn=dqn.FCombined_NN,lr=0.000025,LTL_Game=True,LTL_Net=True):
    """
            Function for training a DQN on an environment specified globally.
            Returns: A trained neural net that takes a gameState and returns a list of Q values (one per action)
            Arguments: 
            - num_frames: Number of frames of Atari game to train on
            - batch_size: Size of minibatch (32 in 2014 paper)
            - gamma: Q-Learning Rate. Indicates how much to weight the new Q value compared to the old in the NN update.
            - replay_memory_size: Number of transitions kept in the replay buffer
            - replay_start_size: Fills up the buffer with random actions until it reaches this many transitions.
            - target_update_frequency: Assign the lag_net to equal the policy_net after this many frames. DQN uses two neural nets. 
                    The lag_net outputs the next Q values. It is kept frozen for 50k frames in the original paper. In each step of stocahstic gradient descent, 
                    the policy net is trained to match the Belleman equation, where the Q value of the next state is taken from lag net.
            - ep_start, ep_end, ep_decay: Manage epsilon, used in epsilon-greedy exploration
            - train_freq: Trains the net once per train_freq frames (4 in the 2014 paper)
            Keyword Arguments:
            - logger_kwargs: Used for logging.
            - starting_model: Saved neural net to start with as policy net.
            - model_save_frequency: Save models during training.
            - n_episodes_plot: Plot performance after this many episodes
            - act_epsilon: Epsilon used for evaluating performance (shockingly 5% in original paper!)
            - nn: Which neural net architecture to use. Options in utils.nnetwork.py
            - lr: Neural net update learning rate, optimal value depends on the optimizer.
            - LTL_Game: Regular atari game or an LTL modified version?
            - LTL_Net: When true, uses a neural net that ouptuts next-Q value and next LTL- automoton state.
    """
    
    log = logger.EpochLogger(**logger_kwargs)
    log.save_config(locals())
    
    # Set up neural nets
    if LTL_Net:
        policy_net = nn(obs_shape[:2],n_actions,n_qstates=4).to(device)
        lag_net = nn(obs_shape[:2],n_actions,n_qstates=4).to(device)
    else:
        policy_net = nn(obs_shape[:2],n_actions,).to(device)
        lag_net = nn(obs_shape[:2],n_actions).to(device)
    
    policy_net.apply(dqn.init_weights)
    if starting_model:
        model_path = f'/data/{starting_model}/pyt_save/model.pt'
        policy_net.load_state_dict(torch.load(cwd+model_path,map_location=device))
    lag_net.load_state_dict(policy_net.state_dict())
    optimizer = Adam(policy_net.parameters(), lr)
    
    # Initialize
    Transition = namedtuple('Transition',('index','state', 'action', 'next_state', 'reward','done','q_state','next_q_state'))
    mem = ReplayMemory(replay_memory_size)
    done = False 			# signal from environment that episode is over
    updated = False			# signal that the nets have been updated at least once
    i_ep, i_frames= 0, 0
    train_freq = 4 			 # update at most every four frames
    epsilon = ep_start
    q_ep = 0.				 # for debugging
    state = env.reset() 	 # first obs comes from starting distribution
    plt_rews,ep_qs,ep_rews = [],[],[]   # for plotting episodes with epsilon=0
    starting_q_state = [1,0,0,0]
    prior_q_state = starting_q_state # initial state of the NBDA.

    # Main loop
    while i_frames < num_frames: # experience transitions associated with the policy.  
        if mem.size < replay_start_size: # initialize replay buffer with random actions
            act = env.action_space.sample() 
        else:
            epsilon = max(ep_end , ep_start - i_frames*(ep_start - ep_end)/eps_decay)
            if LTL_Net:
                act = dqn.get_action(to_tensor([state]),epsilon,policy_net,env,to_tensor([prior_q_state]))
                q_ep += float(get_q(policy_net,to_tensor([state]),to_tensor([prior_q_state]))[act])
            else:
                act =  dqn.get_action(to_tensor([state]),epsilon,policy_net,env)
                q_ep += float(get_q(policy_net,to_tensor([state]))[act])
        
        next_state, rew, done, _ = env.step(act) 
        q_state = ltl.dynamics(prior_q_state, ltl.Label(state.frame(3)))
        next_q_state = ltl.dynamics(q_state, ltl.Label(next_state.frame(3)))
        
        if LTL_Game: # end episode if LTL-automoton rejects LTL sequence.
            if next_q_state == [0,0,1,0] or next_q_state == [0,0,0,1]:
                done = True
                if next_q_state == [0,0,0,1]:
                    rew = 10000
                if next_q_state == [0,0,1,0]:
                    rew = 0
        mem.append(Transition(i_frames,state,act,next_state,rew,done,q_state,next_q_state))
        ep_rews.append(rew)
        i_frames+=1
        
        ok_to_update = (mem.size >= batch_size and mem.size >= replay_start_size and i_frames % train_freq == 0)
        if ok_to_update: # update the Q-function estimator policy_net
            batch = mem.sample(batch_size) # sample a mini-batch
            batch_loss = dqn.train_one_epoch(batch,policy_net=policy_net,target_net=lag_net,optimizer=optimizer,gamma = gamma,LTL_Net=LTL_Net)
            updated = True
            # refresh the target net every C frames.
            if i_frames % target_update_frequency == 0:
                lag_net.load_state_dict(policy_net.state_dict())
            # Save the model every so often
            if i_frames % model_save_frequency==0 : 
                log.pytorch_save(policy_net)
        # Plotting
        if n_episodes_plot and done:
            ep_qs.append(q_ep) 
            if i_ep % n_episodes_plot==0 and epsilon > act_epsilon: # Act out episode
                env_act = wrap_deepmind(make_atari(env_name),episode_life=False, clip_rewards = False,frame_stack=True, scale=True)
                act_ret = act_for_one_episode(net=policy_net,environment=env_act, epsilon=act_epsilon,render=False,LTL_Net=LTL_Net,starting_q_state=starting_q_state,LTL_Game=False)
                plt_rews.append(act_ret)
            else: 
                plt_rews.append(ep_ret)
            if i_ep % n_episodes_plot==0: # Plot results
                fig, qfig, avgfig = plt.figure(), plt.figure(), plt.figure()
                plt_path = f'/data/{exp_name}/'
                plot(fig,plt_rews,"Returns vs Episode",cwd+plt_path)
                plot(avgfig,np.convolve(plt_rews, np.ones(30)/30, mode='valid'),"Running Average Returns",cwd+plt_path)
                plot(qfig,ep_qs,"Total Q Value of Episode",cwd+plt_path)
        if not done:
            state = next_state
            prior_q_state = next_q_state
        if done:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            # Logging
            print(f'Episode finished with length {ep_len}, rew {ep_ret} and total Q estimate {q_ep}. Updated is {updated}.')
            log.log_tabular('Frame',i_frames)
            log.log_tabular('Episode',i_ep)
            log.log_tabular('Final_Q_State', np.argmax(next_q_state))
            log.log_tabular('Epsilon',"%8.3f"%epsilon)
            log.log_tabular('EpRet',ep_ret)
            log.log_tabular('EpLen',ep_len)
            log.log_tabular('Q_ep',"%8.3f"%q_ep)
            log.dump_tabular()
            # Reset
            i_ep +=1
            q_ep, ep_rews = 0., []
            state = env.reset() # env.reset moves to next life when episode_life is True
            prior_q_state = starting_q_state # initial state of the NBDA
    # after training loop
    return policy_net

## Main ##
logger_kwargs = logger.setup_logger_kwargs(exp_name=exp_name) 
log = logger.EpochLogger(**logger_kwargs)
trained_net = train(num_frames = int(2.5e7), batch_size = 32, gamma = .9, replay_memory_size = int(1e6),replay_start_size = 50000, 
                    target_update_frequency=10000,ep_start=10,ep_end=10, eps_decay= int(1e6), train_freq=4, logger_kwargs=logger_kwargs,
                    model_save_frequency=int(1e5),starting_model=None,n_episodes_plot=10,act_epsilon=5,nn=dqn.CNN,lr=0.000025,LTL_Game=True, LTL_Net=False)
log.pytorch_save(trained_net)