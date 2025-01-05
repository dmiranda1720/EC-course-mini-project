'''
Self-Localization code using Hidden Markov Model (HMM)
'''

import numpy as np

def init_probabilities(num_states):
    '''We do not assume any bias on initial position, so we initialize with a random distribution'''
    random_init = np.random.random_sample(size=(num_states,))
    # normalize so all add to one
    random_init_normalized = random_init / np.sum(random_init)
    assert np.allclose(np.sum(random_init_normalized), 1), 'Initial probabilities do not sum to one.'
    return random_init_normalized

def get_state_probabilities(observation, transition_matrix, observation_matrix, observation_history=[]):
    '''
    Localization must return a numpy array of shape (Z,) corresponding to the probability distribution over states
    '''
    #TODO: needs to be implemented 
    return None