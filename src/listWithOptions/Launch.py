#import src.listWithOptions.importEnv as agenv
import src.listWithOptions.Environment as Environment
import src.listWithOptions.Policy as Policy
import src.listWithOptions.QLearning as QLearning
import numpy as np

filename = "data/listWithOptions.txt"
"""
nb_iter = 1000000
#Hyperparameters
alpha = 0.1 #how much we learn (a bit)
gamma = 0.6 #how much importance we give to the futur (lot but not too much)
epsilon = 0.1 #how randomly we move
num_ag = 2 #how many agents
num_row = 5 #map size
num_col = 5 #map size
"""

def TrainingSaveAndTestOne():

	num_ag, num_row, num_col = askDimensions()
	alpha, gamma, epsilon, nb_iter = askHyperparameters()

	env = Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()
	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])

	q_table = QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	QLearning.saveQTableToFile(q_table, num_row, num_col, num_ag, filename)
	print("q_table Saved\nTest:\n")
	QLearning.test(env, q_table, 1)

def TestExistingAgent():

	q_table, num_row, num_col, num_ag = QLearning.getTrainedAgentFromFile(filename)
	print("selected parameters : \ndimensions : "+str(num_row)+"x"+str(num_col)+", "+str(num_ag)+" agents\n")

	env = Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()

	print("Test:\n")
	QLearning.test(env, q_table, 1)
	

def TrainingAndTestOne():

	num_ag, num_row, num_col = askDimensions()
	alpha, gamma, epsilon, nb_iter = askHyperparameters()

	env = Environment.Environment(num_ag, num_row, num_col)
	env.s = 2
	env.reset()

	q_table = np.zeros([env.observation_space.n, env.action_space.n, env.num_agents])

	q_table = QLearning.qLearningTraining_1agent(env, q_table, nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	print("Test:\n")
	QLearning.test(env, q_table, 1)


def askDimensions():
	num_agents = 2
	num_rows = 5
	num_cols = 5

	answer = input('How many agents do you want to place on the map? (int between 2 and 10, max recommanded 4)\n')
	if int(answer)>10:
		num_agents = 10
	elif int(answer)<2:
		num_agents = 2
	else:
		num_agents = int(answer)
	
	answer = input('How many rows do you want the map to be? (int between 2 and 15, max recommanded 6)\n')
	if int(answer)>15:
		num_rows = 15
	elif int(answer)<2:
		num_rows = 2
	else:
		num_rows = int(answer)
	
	answer = input('How many columns do you want the map to be? (int between 2 and 15, max recommanded 6)\n')
	if int(answer)>15:
		num_cols = 15
	elif int(answer)<2:
		num_cols = 2
	else:
		num_cols = int(answer)

	print("selected : "+str(num_agents)+", "+str(num_rows)+", "+str(num_cols)+"\n")
	return num_agents, num_rows, num_cols


def askHyperparameters():
	alpha = 0.1
	gamma = 0.6
	epsilon = 0.1
	num_iterations = 1000000

	answer = input('Select Alpha value : Learning rate (float between 0 and 1, recommanded 0.1)\n')
	if float(answer)>1:
		alpha = 1
	elif float(answer)<0:
		alpha = 0
	else:
		alpha = float(answer)
	
	answer = input('Select Gamma value : Discount factor (float between 0 and 1, recommanded 0.6)\n')
	if float(answer)>1:
		gamma = 1
	elif float(answer)<0:
		gamma = 0
	else:
		gamma = float(answer)
	
	answer = input('Select Epsilon value : Random factor (float between 0 and 1, recommanded 0.1)\n')
	if float(answer)>1:
		epsilon = 1
	elif float(answer)<0:
		epsilon = 0
	else:
		epsilon = float(answer)

	answer = input('How many times do you want to train your agents? (int, power of 10, max recommanded 1 000 000)\n')
	num_iterations = int(answer)

	print("selected : "+str(alpha)+", "+str(gamma)+", "+str(epsilon)+", "+str(num_iterations)+"\n")
	return alpha, gamma, epsilon, num_iterations