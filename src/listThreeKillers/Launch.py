import src.listThreeKillers.importEnv as agenv
import numpy as np

nb_iter = 1000000
#Hyperparameters
alpha = 0.1 #how much we learn (a bit)
gamma = 0.6 #how much importance we give to the futur (lot but not too much)
epsilon = 0.1 #how randomly we move
num_ag = 3 #how many agents
num_row = 7 #map size
num_col = 7 #map size

def TrainingSaveAndTestOne():
	global nb_iter
	global num_ag 
	global num_row 
	global num_col 
	global alpha 
	global gamma 
	global epsilon 

	env = agenv.Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()
	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])

	q_table = agenv.QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	agenv.QLearning.saveQTableToFile(q_table, agenv.filename)
	print("q_table Saved\nTest:\n")
	agenv.QLearning.test(env, q_table, 1)

def TestExistingAgent():
	global num_ag 
	global num_row 
	global num_col 

	env = agenv.Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()
	"""
	ar, ac = env.decode(env.s)
	env.render([False for _ in range(num_ag)], ar, ac, [0 for _ in range(num_ag)])
	"""
	q_table = agenv.QLearning.getTrainedAgentFromFile(agenv.filename)

	print("Test:\n")
	agenv.QLearning.test(env, q_table, 1)
	

def TrainingAndTestOne():
	global nb_iter
	global num_ag 
	global num_row 
	global num_col 
	global alpha 
	global gamma 
	global epsilon 

	env = agenv.Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()

	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])

	q_table = agenv.QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	print("Test:\n")
	agenv.QLearning.test(env, q_table, 1)