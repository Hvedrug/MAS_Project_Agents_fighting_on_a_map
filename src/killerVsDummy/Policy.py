import numpy as np

def bestFirstPolicy(q_table, state):
	# take a table of dimension observation_space x action_space and int(state)
	# each cell of it contains a value that give an idea of how good doing an action is from a state
	# this function take a state and return the best action possible
	return np.argmax(q_table[state])

def randomPolicy(env):
	# return a random possible action from the environment
	return env.action_space.sample()

def randomMovingPolicy(env):
	action = env.action_space.sample()
	if action>=4:
		action = action-4
	return action 

def randomShootingPolicy(env):
	action = env.action_space.sample()
	if action <4:
		action = action+4
	return action 

