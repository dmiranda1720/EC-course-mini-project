'''
Miscellaneous functions used in developing/testing this project
'''

import warnings
import numpy as np
import matplotlib.pylab as plt

warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium.envs.registration import register

import environment
import policy

def gaussian_2d_array(shape, sigma=1.0, center=(0,0)):
	# create a meshgrid of x and y coordinates
	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
	# calculate the distance a center provided
	dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	# generate the Gaussian values
	gaussian = np.exp(-(dist**2) / (2 * sigma**2))
	# normalize the values to sum to 1
	gaussian /= np.sum(gaussian)
	return gaussian

def plot_gaussian_grid(grid, cmap='plasma_r'):
	plt.imshow(np.flipud(grid.T), cmap=cmap, interpolation='nearest')
	plt.colorbar()
	plt.title("2D Gaussian Distribution")
	plt.show()

def load_env(gridworld_dimensions=20, obstacle_ratio=0.4):
	# validate gridworld dimensions per assignment requirements
	assert gridworld_dimensions in set([20,30]), 'gridworld_dimensions must be 20 or 30. Received: {}'.format(gridworld_dimensions)
	assert 0.0 <= obstacle_ratio <= 1.0, 'obstacle_ratio value must be in range [0,1]'

	env_config = {
	'name': 'GridWorld-v0',
	'config': (environment.GridWorldEnv, {'N':gridworld_dimensions, 'obstacle_ratio':obstacle_ratio})
	}
	
	register(id=env_config['name'], entry_point=env_config['config'][0], kwargs=env_config['config'][1])
	env = gym.make(env_config['name'], disable_env_checker=True)
	return env

def simulate_dummy_scenario(gridworld_dimensions=20, obstacle_ratio=0.4, 
							high_uncertainty_duration=0.5, sigma_range=(50.0,2.0), max_steps=100, 
							save_to_path='figures/sample.gif'):
	'''
	Simulate a scenario with a Gaussian probability for localization.
	Used to validate system and visualization.
	
	Parameters:
	- high_uncertainty_duration: how long to maintain a high standard deviation (std)
	- sigma_range: std bounds for scenario
	- max_steps: max number of steps in episode, used to decide how long to maintain high/low uncertainty

	'''    

	
	max_sigma, min_sigma = sigma_range
	# generate a sequence of decreasing sigmas with a sustained region of high-certainty
	sigmas = np.linspace(max_sigma, min_sigma, num=int(max_steps*high_uncertainty_duration))
	sigmas = np.concatenate([sigmas, min_sigma*np.ones(shape=(int(max_steps*(1-high_uncertainty_duration)),))])

	# load environment 
	env = load_env(gridworld_dimensions, obstacle_ratio)
	
	# load robot motion controller
	robot = policy.RandomActionAgent()
	
	# generate a new world
	observation, T, O, info = env.reset()
	
	# init state probabilities 
	init_probs = gaussian_2d_array(info['grid_shape'], sigmas[0], center=info['agent_position']).flatten()
	
	# generate initial frame
	frame = env.render(state_probabilities=init_probs)
	frames = [frame]
	
	# simulate multiple steps
	for t in range(1,max_steps):
		# robot observes and takes an action
		action = robot.act(observation)
		# step environment with action and receive new observations
		observation, _, _, _, info = env.step(action=action)
		# compute new probabilities
		probs_t = gaussian_2d_array(info['grid_shape'], sigmas[t], center=info['agent_position']).flatten()
		# generate new frame
		frame = env.render(state_probabilities=probs_t)
		frames.append(frame)
	
	# save frames to gif
	environment.save_gif(frames, save_to_path=save_to_path)
	return

