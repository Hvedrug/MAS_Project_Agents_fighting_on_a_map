import src.listKillerVsRunner.importEnv as agenv
import numpy as np

nb_iter = 1000000
#Hyperparameters
alpha = 0.1 #how much we learn (a bit)
gamma = 0.6 #how much importance we give to the futur (lot but not too much)
epsilon = 0.1 #how randomly we move

def TrainingSaveAndTestOne():
	global nb_iter
	env = agenv.Environment.Environment()
	env.s = 2
	env.reset()
	#q_table = np.zeros([env.observation_space.n, env.action_space.n])
	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])

	#Hyperparameters
	global alpha #how much we learn (a bit)
	global gamma #how much importance we give to the futur (lot but not too much)
	global epsilon #how randomly we move

	q_table = agenv.QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	agenv.QLearning.saveQTableToFile(q_table, agenv.filename)
	print("q_table Saved\nTest:\n")
	agenv.QLearning.test(env, q_table, 1)

def TestExistingAgent():
	env = agenv.Environment.Environment()
	env.s = 2
	env.reset()
	q_table = agenv.QLearning.getTrainedAgentFromFile(agenv.filename)

	#Hyperparameters
	global alpha #how much we learn (a bit)
	global gamma #how much importance we give to the futur (lot but not too much)
	global epsilon #how randomly we move

	print("Test:\n")
	agenv.QLearning.test(env, q_table, 1)

def TrainingAndTestOne():
	global nb_iter
	env = agenv.Environment.Environment()
	env.s = 2
	env.reset()
	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])
	#q_table = np.zeros([env.observation_space.n, env.action_space.n])

	#Hyperparameters
	global alpha #how much we learn (a bit)
	global gamma #how much importance we give to the futur (lot but not too much)
	global epsilon #how randomly we move

	q_table = agenv.QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	print("Test:\n")
	agenv.QLearning.test(env, q_table, 1)