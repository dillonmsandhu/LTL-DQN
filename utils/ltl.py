import numpy as np

def Label(s):
    """Takes a lazy-frame state s and returns a set in 2^AP called the label"""
    def blue(s):
        """
        Take in a pre-processed frame s and determine if there are any blue blocks.
        Uses the final frame in the stack of 4 pre-processed "frames" that's considered a state.
        """
        return np.any(s[35:37,] == 1/3) # blue in greyscale is 1/3 
    def other_blocks_full(s):
        """takes a lazyframe state s, and returns whether all the non-blue blocks are full"""
        return np.all(s[25:35,]!=0)
    def all_empty(s):
        """technically we wanted an agent that can win, but winning requires clearing two screens of blocks, 
        for a score of 448x2 = 896. However, not many agents can do this:
        https://paperswithcode.com/sota/atari-games-on-atari-2600-breakout
        Implimenting an all_empty function instead - correlating to one round
        """
        return np.all(s[25:37,5:-5]==0)
    # def won(total_rew):
#       return total_rew == 896

    # def cleared_screen(total_rew):
    #     return total_rew == 448
    
    label = set()
    if blue(s):
        label.add('Blue')
    if other_blocks_full(s):
        label.add('Rest_Full')
    if all_empty(s):
        label.add('Cleared')
    return label

def dynamics(prior_q_state,label):
    """The dynamics function (little-delta) representing the NDBA of our LTL-criteria"""
    if prior_q_state == [1,0,0,0]: #"q0"
        if 'Blue' in label:
            if 'Rest_Full' in label:
                return [1,0,0,0] # "q0"
            else: return [0,1,0,0] #"q1"
        if not 'Blue' in label and "Rest_Full" in label:
            return [0,0,1,0] #"q2"
        else: raise Exception(f'Invalid transition from {prior_q_state} with label {label}')
    if prior_q_state == [0,1,0,0]: #"q1" - terminal state
        return prior_q_state    
    if prior_q_state ==  [0,0,1,0]: #"q2"
        if 'Cleared' in label:
            return [0,0,0,1] #"q3"
        else: return [0,0,1,0] #"q2"
    if prior_q_state == [0,0,0,1]: #"q3" - terminal state
        return prior_q_state
