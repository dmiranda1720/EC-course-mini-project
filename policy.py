import random 
from environment import Actions, Observations

class RandomActionAgent(object):
	def __init__(self):
		self.valid_actions = {action.name for action in Actions}
	
	def act(self, observation):
		
		# get the set of actions unavailable given the observation
		actions_not_available = {
			Actions(Observations.get_direction_by_index(obs_idx)).name 
			for obs_idx, obs in enumerate(observation)
			if obs==1
		}

		# get the names of the actions available by doing set subtraction
		actions_available = self.valid_actions - actions_not_available
		# only keep the actions available in Action(Enum) type
		actions_available = [Actions[name] for name in actions_available]
		return  random.choice(actions_available).name