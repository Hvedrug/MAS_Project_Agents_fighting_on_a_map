import numpy as np
import src.killerVsDummy.Policy as Policy
from IPython.display import clear_output
from time import sleep

def test(env, q_table, num_episodes=10):
	total_epochs, total_penalties = 0, 0
	all_epochs = 0
	penalties = 0
	for _ in range(num_episodes):
		state = env.reset()
		epochs, penalties, reward = 0, 0, 0
		done = False
		a1r, a1c, a2r, a2c = env.decode(state)
		clear_output(wait=True)
		print("State: " +str(state))
		print("Action: none")
		print("Reward: " +str(reward))
		env.render(done, a1r, a1c, a2r, a2c)
		clear_output(wait=True)
		sleep(.5)

		while not done:
			action = Policy.bestFirstPolicy(q_table, state)
			#choose the best action computed before
			state, reward, done, info = env.step(action)
			# perform the action choosen

			if reward == -10:
				penalties += 1
			epochs += 1
			
			a1r, a1c, a2r, a2c = env.decode(state)
			clear_output(wait=True)
			print("State: " +str(state))
			print("Action: " +str(action))
			print("Reward: " +str(reward))
			env.render(done, a1r, a1c, a2r, a2c, action)
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

		epochs, penalties, reward = 0, 0, 0
		done = False

		while not done:
			if np.random.uniform(0, 1)<epsilon:
				action = Policy.randomPolicy(env)
				#if less than epsilon, pick a random action
			else:
				action = Policy.bestFirstPolicy(q_table, state)
				#choose the best immediate solution
			#perform the selected action on the env
			next_state, reward, done, info = env.step(action)
			#update Q_table with what we get
			old_value = q_table[state, action]
			next_max = np.max(q_table[next_state])
			q_table[state, action] = (1-alpha)*old_value + alpha*(reward+gamma*next_max)
			# the value is computed with how much importance you give to the past (1-alpha) and how much you give to the futur + alpha*(...)

			if reward == -10:
				penalties += 1

			state = next_state
			epochs += 1
		if i%10000 == 0:
			print(f"Episode: {i}")
	return q_table

def getTrainedAgentFromFile(filename):
	f = open(filename, "r")
	data = f.read()
	data = data.split(';')
	for i in range(len(data)):
		data[i] = data[i].split(',')
	q_table = np.zeros([len(data), len(data[0])])
	for j in range(len(data)):
		for k in range(len(data[j])):
			q_table[j,k] = float(data[j][k])
		
	return q_table

def saveQTableToFile(q_table, filename):
	f = open(filename, "w")
	data = ""
	for i in range(len(q_table)):
		for j in range(len(q_table[i])):
			data+=str(q_table[i][j])
			data+=","
		data = data[:-1]
		data+=";"
	data = data[:-1]
	f.write(data)