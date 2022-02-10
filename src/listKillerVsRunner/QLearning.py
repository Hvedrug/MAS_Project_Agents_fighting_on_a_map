import numpy as np
import src.listKillerVsRunner.Policy as Policy
from IPython.display import clear_output
from time import sleep

def test(env, q_table, num_episodes=10):
	total_epochs, total_penalties = 0, 0
	all_epochs = 0
	penalties = 0
	for _ in range(num_episodes):
		state = env.reset()
		epochs, penalties, reward = 0, 0, 0
		dones = [False, False]
		done = False
		rows, cols = env.decode(state)
		clear_output(wait=True)
		print("State: " +str(state))
		print("Action: none")
		print("Reward: " +str(reward))
		env.render(dones, rows, cols, [])
		clear_output(wait=True)
		sleep(.5)
		actions = [0,0]

		while not done:
			actions[0] = Policy.bestFirstPolicy(q_table, state, 0)

			#random move from agent 2
			actions[1] = Policy.randomMovingPolicy(env)

			
			#choose the best action computed before
			state, rewards, dones, infos = env.step(actions)
			# perform the action choosen

			if -10 in rewards:
				penalties += 1

			if True in dones:
				done = True

			epochs += 1
			

			rows, cols = env.decode(state)
			clear_output(wait=True)
			print("State: " +str(state))
			print("Action: " +str(actions))
			print("Reward: " +str(rewards))
			env.render(dones, rows, cols, actions)
			clear_output(wait=True)
			sleep(.5)
			
		total_penalties += penalties
		total_epochs += all_epochs

	clear_output(wait=True)
	print(f"Results after {num_episodes} episodes :")
	print(f"Average timesteps per episode: {total_epochs/num_episodes}")
	print(f"Average penalties per episode: {total_penalties/num_episodes}")


def qLearningTraining_1agent(env, q_table, num_iteration, alpha=0.1, gamma=0.6, epsilon=0.1):
	# for plotting metrics
	all_epochs = []
	all_penalties = []

	for i in range(1, num_iteration):
		state = env.reset()

		epochs, penalties, reward = 0, 0, [0,0]
		done = False
		actions = [0,0]

		while not done:
			if np.random.uniform(0, 1)<epsilon:
				actions[0] = Policy.randomPolicy(env)
				#if less than epsilon, pick a random action
			else:
				actions[0] = Policy.bestFirstPolicy(q_table, state, 0)
				#choose the best immediate solution

			#random move from agent 2
			actions[1] = Policy.randomMovingPolicy(env)

			#perform the selected action by agent 1 on the env
			next_state, rewards, dones, infos = env.step(actions)
			#update Q_table with what we get
			old_value = q_table[state, actions[0], 0]
			next_max = np.max(q_table[next_state])
			q_table[state, actions[0]] = (1-alpha)*old_value + alpha*(rewards[0]+gamma*next_max)
			# the value is computed with how much importance you give to the past (1-alpha) and how much you give to the futur + alpha*(...)

			if -10 in rewards:
				penalties += 1

			if True in dones:
				done = True

			state = next_state
			epochs += 1
		if i%1000 == 0:
			print(f"Episode: {i//1000}")
	return q_table

def getTrainedAgentFromFile(filename):
	f = open(filename, "r")
	data = f.read()
	data = data.split(';')
	for i in range(len(data)):
		data[i] = data[i].split(',')
	for i in range(len(data)):
		for j in range(len(data[0])):
			data[i][j] = data[i][j].split('!')
	q_table = np.zeros([len(data), len(data[0]), len(data[0][0])])
	for j in range(len(data)):
		for k in range(len(data[j])):
			for l in range(len(data[j][k])):
				q_table[j,k,l] = float(data[j][k][l])
		
	return q_table

def saveQTableToFile(q_table, filename):
	f = open(filename, "w")
	data = ""
	for i in range(len(q_table)):
		for j in range(len(q_table[i])):
			for k in range(len(q_table[i][j])):
				data+=str(q_table[i][j][k])
				data+="!"
			data = data[:-1]
			data+=","
		data = data[:-1]
		data+=";"
	data = data[:-1]
	f.write(data)