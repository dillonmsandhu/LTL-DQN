import numpy as np
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self,sizes,n_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels = 32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels = 64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 64,kernel_size=3,stride=1)
        
        # determine the size of the output from a 2d convolutional layer
        def output_len(input_len,kernel_l,stride,padding=0):
            return (input_len - kernel_l + 2*padding)//stride + 1
        
        convw = output_len(output_len(output_len(sizes[1],8,4),4,2),3,1)
        convh = output_len(output_len(output_len(sizes[0],8,4),4,2),3,1)
        fc_size = convw*convh*64
        self.fc = nn.Linear(fc_size, n_actions)

    def forward(self, x):
        # Forward pass
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.fc(torch.reshape(x,(x.shape[0],-1))))
        return x

def init_weights(m):
    print(f'initializing weights for a layer of type {type(m)}')
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
    else:
        print(type(m))
        assert "Check the layer type"

def to_tensor(batch):
    """converts a list to tensor for input into the nets"""
    batch_array = np.array(batch).astype('float32')
    # rearanges atari states into torch format of (index x chanel x row x col)
    if batch_array.ndim == 4:
        batch_array = batch_array.transpose((0,3,1,2))
    return torch.as_tensor(batch_array).to(device)

# make action selection function (outputs int actions, sampled from policy)
def get_action(state,epsilon,net,env,q_state=None):
    """epsilon greedy"""
    if random.uniform(0., 100.) <= epsilon:
        return torch.as_tensor(env.action_space.sample())
    try:
        Q_hat = net(state,q_state)
    except:
        Q_hat = net(state)
    return torch.argmax(Q_hat).to(device)

def get_q(network,states,q_states=None):
    # assert torch.is_tensor(states)
    try:
        return torch.squeeze(network(states,q_states))
    except:
        return torch.squeeze(network(states))

def update(net,optimizer,y_hat,y_star):
    """used to update the policy net"""
    optimizer.zero_grad()
    loss_fn = nn.SmoothL1Loss() #Huber
    # loss_fn = nn.MSELoss()
    batch_loss = loss_fn(y_hat, y_star)
    batch_loss.backward()
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)
        optimizer.step()
    return batch_loss

def train_one_epoch(batch,policy_net,target_net,optimizer,gamma,LTL_Net = True):
    state_batch = to_tensor([trans.state for trans in batch])
    next_state_batch = to_tensor([trans.next_state for trans in batch ])
    action_batch = [trans.action for trans in batch]
    reward_batch = torch.tensor([trans.reward for trans in batch]).to(device)
    final_state_mask = [i for i,trans in enumerate(batch) if trans.done]
    if LTL_Net:
        q_state_batch = torch.tensor([trans.q_state for trans in batch]).float().to(device)
        next_q_state_batch = torch.tensor([trans.next_q_state for trans in batch]).float().to(device)
        Q_hats = get_q(policy_net,state_batch,q_state_batch)
        Q_star_next, _ = torch.max(get_q(target_net,next_state_batch,next_q_state_batch),1) 
    if not LTL_Net:
        Q_hats = get_q(policy_net,state_batch)
        Q_star_next, _ = torch.max(get_q(target_net,next_state_batch),1) 
    # get NN's estimate of Q
    Q_hats = Q_hats.gather(1,torch.unsqueeze(torch.tensor(action_batch).to(device),1))
    Q_hats = Q_hats.squeeze().to(device)
    # get TD(1)-estimate of Q        
    Q_star_next[final_state_mask] = 0.
    Q_star_next *= gamma 
    Q_star_next += reward_batch
    Q_star_next.to(device)
    batch_loss = update(policy_net,optimizer,y_hat=Q_hats, y_star=Q_star_next)
    return batch_loss

## LTL-equipted Neural Nets. Take two inputs:

class Directly_Combined_NN(nn.Module):
    """This neural net takes two inputs: 
        1) the lazy-frames state(s) as a tensor with dimensions (index, channel, len, width) 
        2) the one hot endcoding of the q-state(s) as a tensor with dimensions (index,num_q_states)
    and ouptus a vector of lenth n_actions, with a Q-estimate for each action.
    The architecture combines the output of the convolutional layers directly with the one-hot encoding of the
    q-states
    """
    def __init__(self,sizes,n_actions,n_qstates):
        super(Directly_Combined_NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels = 32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels = 64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 64,kernel_size=3,stride=1)
        # determine the size of the output from a 2d convolutional layer
        def output_len(input_len,kernel_l,stride,padding=0):
            return (input_len - kernel_l + 2*padding)//stride + 1
        convw = output_len(output_len(output_len(sizes[1],8,4),4,2),3,1)
        convh = output_len(output_len(output_len(sizes[0],8,4),4,2),3,1)
        fc_size = convw*convh*64
        self.fc = nn.Linear(fc_size+n_qstates, n_actions)

    def forward(self, input1, input2):
        # Forward pass
        c = torch.relu(self.conv1(input1))
        c = torch.relu(self.conv2(c))
        c = torch.relu(self.conv3(c))
        combined = torch.cat((torch.reshape(c,(c.shape[0],-1)), 
                                    torch.reshape(input2,(input2.shape[0],-1))), 
                            dim=1)
        c = torch.relu(self.fc(combined))
        return c
    
class FCombined_NN(nn.Module):
    """This neural net takes two inputs: 
        1) the lazy-frames state(s) as a tensor with dimensions (index, channel, len, width) 
        2) the one hot endcoding of the q-state(s) as a tensor with dimensions (index,num_q_states)
    and ouptus a vector of lenth n_actions, with a Q-estimate for each action.
    
    The architecture passes the one-hot encoding of the q-state through a fully connected layer with 32 outputs before
    concatenating and sending to the final fully connected layer.
    """
    def __init__(self,sizes,n_actions,n_qstates):
        super(FCombined_NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels = 32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels = 64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 64,kernel_size=3,stride=1)
        # determine the size of the output from a 2d convolutional layer
        def output_len(input_len,kernel_l,stride,padding=0):
            return (input_len - kernel_l + 2*padding)//stride + 1
        convw = output_len(output_len(output_len(sizes[1],8,4),4,2),3,1)
        convh = output_len(output_len(output_len(sizes[0],8,4),4,2),3,1)
        fc_size = convw*convh*64
        self.fc = nn.Linear(fc_size+32, n_actions)
        self.input2_fc = nn.Linear(n_qstates,32)

    def forward(self, input1, input2):
        # Forward pass
        c = torch.relu(self.conv1(input1))
        c = torch.relu(self.conv2(c))
        c = torch.relu(self.conv3(c))
        f = torch.relu(self.input2_fc(input2))
        combined = torch.cat((torch.reshape(c,(c.shape[0],-1)), 
                                torch.reshape(f,(f.shape[0],-1)))
                            , dim=1)
        combined = torch.relu(self.fc(combined))
        return combined