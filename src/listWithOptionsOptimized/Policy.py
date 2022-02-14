import numpy as np
import src.listWithOptionsOptimized.Data as Data

def bestFirstPolicy(state):
	# take a table of dimension observation_space x action_space and int(state)
	# each cell of it contains a value that give an idea of how good doing an action is from a state
	# this function take a state and return the best action possible
	
	tab = Data.q_table[state]
	tab = tab.transpose()
	#res = tab[agent]
	res = np.argmax(tab)
	return res

def randomPolicy():
	# return a random possible action from the environment
	return Data.env.action_space.sample()

def randomMovingPolicy():
	action = Data.env.action_space.sample()
	if action>=4:
		action = action-4
	return action 

def randomShootingPolicy():
	action = Data.env.action_space.sample()
	if action <4:
		action = action+4
	return action 

