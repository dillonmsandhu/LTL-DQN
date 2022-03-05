# LTL is Always -All_Blue -> (rest_full ^ All_But_One_Blue) ^ Eventually Win
# In english: remove a non-blue block after removing the first blue block 
import numpy as np

# Alternative LTL: 
def Label(s):
    """Takes a lazy-frame state s and returns a set in 2^AP called the label"""
    def at_least_p_blue(s,p=.5):
        """At least p percent of initially blue pixels are blue"""
        blue_pixels = np.sum(s[35:37,5:-5] == 1/3)
        return blue_pixels / 148 >= p

    def other_blocks_full(s):
        """Returns whether all the non-blue blocks are full"""
        return np.all(s[25:35,]!=0)
    
    def all_empty(s):
        return np.all(s[25:37,5:-5]==0)

    label = set()
    if at_least_p_blue(s,.85):
        label.add('At_Least_91_Blue')
    if other_blocks_full(s):
        label.add('Rest_Full')
    if all_empty(s):
        label.add('Cleared')
    return label


def dynamics(prior_q_state,label):
    """The dynamics function (little-delta) representing the NDBA of our LTL-criteria"""
    
    if prior_q_state == [1,0,0,0]: #"q0 can go to either q2 (terminal) or q1"
        if 'At_Least_91_Blue' in label and "Rest_Full" in label:
            return [1,0,0,0] #" stays in q0

        if 'At_Least_91_Blue' and not('Rest_Full' in label):
            return [0,1,0,0] #q0 followed by more than 2 blue hit and another block hit -> q1) 

        if not ('At_Least_91_Blue' in label) and 'Rest_Full' in label:
            return [0,0,1,0] # "q0 followed by more than 2 blue hit and still rest full -> q2"
        
        else: raise Exception(f'Invalid transition from {prior_q_state} with label {label}')
    
    if prior_q_state == [0,0,1,0]: #"q2" - terminal state
        return prior_q_state    
    
    if prior_q_state ==  [0,1,0,0]: #"q1"
        if 'Cleared' in label:
            return [0,0,0,1] #"q3"
        else: return [0,1,0,0] #"q1"
    
    if prior_q_state == [0,0,0,1]: #"q3" - terminal state
        return prior_q_state