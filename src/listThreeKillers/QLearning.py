import numpy as np
import src.listThreeKillers.Policy as Policy
from IPython.display import clear_output
from time import sleep

def test(env, q_table, num_episodes=10):
	total_epochs, total_penalties = 0, 0
	all_epochs = 0
	penalties = 0
	for _ in range(num_episodes):
		state = env.reset()
		epochs, penalties, rewards = 0, 0, [0 for _ in range(env.num_agents)]
		dones = [False for _ in range(env.num_agents)]
		done = False
		rows, cols = env.decode(state)
		clear_output(wait=True)
		print("State: " +str(state))
		print("Action: none")
		print("Reward: " +str(rewards))
		print("isAgentDead: "+str(env.isAgentDead))
		env.render(dones, rows, cols, [])
		clear_output(wait=True)
		sleep(.5)
		actions = [0 for _ in range(env.num_agents)]

		while not done:

			for i in range(env.num_agents):
				actions[i] = Policy.bestFirstPolicy(q_table, state, i)
				
			if actions == env.lastActions:
				#if looping on the same action then random moves for one
				randValue = np.random.uniform(0, 1)

				for j in range(env.num_agents):
					if randValue >= j*(1/env.num_agents) and randValue < (j+1)*(1/env.num_agents):
						actions[j] = Policy.randomPolicy(env)

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
			print("isAgentDead: "+str(env.isAgentDead))
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

		epochs, penalties, reward = 0, 0, [0 for _ in range(env.num_agents)]
		done = False
		actions = [0 for _ in range(env.num_agents)]

		while not done:
			for j in range(env.num_agents):

				if np.random.uniform(0, 1)>=epsilon:
					actions[j] = Policy.randomPolicy(env)
				else:
					actions[j] = Policy.bestFirstPolicy(q_table, state, j)

			#perform the selected action by agent 1 on the env
			next_state, rewards, dones, infos = env.step(actions)

			for k in range(env.num_agents):
				old_value = q_table[state, actions[k], k]
				next_max = Policy.bestFirstPolicy(q_table, next_state, k) #np.max(q_table[next_state])
				q_table[state, actions[k], k] = (1-alpha)*old_value + alpha*(rewards[k]+gamma*next_max)
				# the value is computed with how much importance you give to the past (1-alpha) and how much you give to the futur + alpha*(...)

			if -10 in rewards:
				penalties += 1

			if True in dones:
				done = True

			state = next_state
			epochs += 1
		if i%1000 == 0:
			print(f"Episode: {i//1000}/{num_iteration//1000}")
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
	f.close()
		
	return q_table

def saveQTableToFile(q_table, filename):
	f = open(filename, "w")
	data = ""
	data = ';'.join([','.join(['!'.join([str(x) for x in col]) for col in row]) for row in q_table])
	f.write(data)
	f.close()
