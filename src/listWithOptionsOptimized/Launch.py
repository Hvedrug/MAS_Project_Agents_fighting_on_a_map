import src.listWithOptionsOptimized.Environment as Environment
import src.listWithOptionsOptimized.Policy as Policy
import src.listWithOptionsOptimized.QLearning as QLearning
import src.listWithOptionsOptimized.Data as Data
import numpy as np


def TrainingSaveAndTestOne():

	num_ag, num_row, num_col = askDimensions()
	alpha, gamma, epsilon, nb_iter = askHyperparameters()

	Data.env = Environment.Environment(num_ag, num_row, num_col)
	Data.env.s = 2
	Data.env.reset()
	Data.q_table = np.zeros([Data.env.observation_space.n, Data.env.action_space.n])

	QLearning.qLearningTraining_1agent(nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	QLearning.saveQTableToFile(num_row, num_col, num_ag, Data.filename)
	print("q_table Saved\nTest:\n")
	QLearning.test(1)

def TestExistingAgent():

	num_row, num_col, num_ag = QLearning.getTrainedAgentFromFile(Data.filename)
	print("selected parameters : \ndimensions : "+str(num_row)+"x"+str(num_col)+", "+str(num_ag)+" agents\n")

	Data.env = Environment.Environment(num_ag, num_row, num_col)
	Data.env.s = 2
	Data.env.reset()

	print("Test:\n")
	QLearning.test(1)
	

def TrainingAndTestOne():

	num_ag, num_row, num_col = askDimensions()
	alpha, gamma, epsilon, nb_iter = askHyperparameters()

	Data.env = Environment.Environment(num_ag, num_row, num_col)
	Data.env.s = 2
	Data.env.reset()

	Data.q_table = np.zeros([Data.env.observation_space.n, Data.env.action_space.n])

	QLearning.qLearningTraining_1agent(nb_iter, alpha, gamma, epsilon)
	print("Training finished")
	print("Test:\n")
	QLearning.test(1)


def askDimensions():
	num_agents = 2
	num_rows = 5
	num_cols = 5

	answer = input('How many agents do you want to place on the map? (int between 2 and 10, max recommanded 4)\n')
	if int(answer)>10:
		num_agents = 10
	elif int(answer)<2:
		num_agents = 2
	elif int(answer)<10 and int(answer)>2:
		num_agents = int(answer)
	
	answer = input('How many rows do you want the map to be? (int between 2 and 15, max recommanded 6)\n')
	if int(answer)>15:
		num_rows = 15
	elif int(answer)<2:
		num_rows = 2
	elif int(answer)<15 and int(answer)>2:
		num_rows = int(answer)
	
	answer = input('How many columns do you want the map to be? (int between 2 and 15, max recommanded 6)\n')
	if int(answer)>15:
		num_cols = 15
	elif int(answer)<2:
		num_cols = 2
	elif int(answer)<15 and int(answer)>2:
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
	elif float(answer)<1 and float(answer)>0:
		alpha = float(answer)
	
	answer = input('Select Gamma value : Discount factor (float between 0 and 1, recommanded 0.6)\n')
	if float(answer)>1:
		gamma = 1
	elif float(answer)<0:
		gamma = 0
	elif float(answer)<1 and float(answer)>0:
		gamma = float(answer)
	
	answer = input('Select Epsilon value : Random factor (float between 0 and 1, recommanded 0.1)\n')
	if float(answer)>1:
		epsilon = 1
	elif float(answer)<0:
		epsilon = 0
	elif float(answer)<1 and float(answer)>0:
		epsilon = float(answer)

	answer = input('How many times do you want to train your agents? (int, power of 10, max recommanded 1 000 000)\n')
	num_iterations = int(answer)

	print("selected : "+str(alpha)+", "+str(gamma)+", "+str(epsilon)+", "+str(num_iterations)+"\n")
	return alpha, gamma, epsilon, num_iterations