import numpy as np
import src.listWithOptionsOptimized.Policy as Policy
import src.listWithOptionsOptimized.Data as Data
from IPython.display import clear_output
from time import sleep

def test(num_episodes=10):
	total_epochs, total_penalties = 0, 0
	all_epochs = 0
	penalties = 0
	for _ in range(num_episodes):
		state = Data.env.reset()
		epochs, penalties, rewards = 0, 0, [0 for _ in range(Data.env.num_agents)]
		done = False
		rows, cols = Data.env.decode(state)
		clear_output(wait=True)
		print("State: " +str(state))
		print("Action: none")
		print("Reward: " +str(rewards))
		print("isAgentDead: "+str(Data.env.isAgentDead))
		print("done: "+str(done))
		Data.env.render(rows, cols, [])
		clear_output(wait=True)
		sleep(.5)
		actions = [0 for _ in range(Data.env.num_agents)]

		while not done:

			for i in range(Data.env.num_agents):
				#actions[i] = Policy.bestFirstPolicy(q_table, state)
				actions[i] = extractActionFromSubState(state, i)
				
			if actions == Data.env.lastActions:
				#if looping on the same action then random moves for one
				randValue = np.random.uniform(0, 1)

				for j in range(Data.env.num_agents):
					if randValue >= j*(1/Data.env.num_agents) and randValue < (j+1)*(1/Data.env.num_agents):
						actions[j] = Policy.randomPolicy()

			#choose the best action computed before
			state, rewards, done, infos = Data.env.step(actions)
			# perform the action choosen

			if -10 in rewards:
				penalties += 1

			epochs += 1
			

			rows, cols = Data.env.decode(state)
			clear_output(wait=True)
			print("State: " +str(state))
			print("Action: " +str(actions))
			print("Reward: " +str(rewards))
			print("isAgentDead: "+str(Data.env.isAgentDead))
			print("done: "+str(done))
			Data.env.render(rows, cols, actions)
			clear_output(wait=True)
			sleep(.5)
			
		total_penalties += penalties
		total_epochs += all_epochs

	clear_output(wait=True)
	print(f"Results after {num_episodes} episodes :")
	print(f"Average timesteps per episode: {total_epochs/num_episodes}")
	print(f"Average penalties per episode: {total_penalties/num_episodes}")


def qLearningTraining_1agent(num_iteration, alpha=0.1, gamma=0.6, epsilon=0.1):
	# for plotting metrics
	all_epochs = []
	all_penalties = []

	for i in range(1, num_iteration):
		state = Data.env.reset()

		epochs, penalties, reward = 0, 0, [0 for _ in range(Data.env.num_agents)]
		done = False
		actions = [0 for _ in range(Data.env.num_agents)]

		while not done:
			for j in range(Data.env.num_agents):

				if np.random.uniform(0, 1)>=epsilon:
					actions[j] = Policy.randomPolicy()
				else:
					actions[j] = Policy.bestFirstPolicy(state)

			#perform the selected action by agent 1 on the env
			next_state, rewards, done, infos = Data.env.step(actions)

			for k in range(Data.env.num_agents):
				if Data.env.isAgentDead[k]==False:
					old_value = Data.q_table[state, actions[k]]
					next_max = Policy.bestFirstPolicy(next_state) #np.max(q_table[next_state])
					Data.q_table[state, actions[k]] = (1-alpha)*old_value + alpha*(rewards[k]+gamma*next_max)
					# the value is computed with how much importance you give to the past (1-alpha) and how much you give to the futur + alpha*(...)

			if -10 in rewards:
				penalties += 1

			state = next_state
			epochs += 1
		if i%1000 == 0:
			print(f"Episode: {i//1000}/{num_iteration//1000}")

def getTrainedAgentFromFile(filename):
	f = open(filename, "r")
	fileContent = f.read()
	fileContent = fileContent.split('?')
	info = fileContent[1]
	data = fileContent[0]
	data = data.split(';')
	for i in range(len(data)):
		data[i] = data[i].split(',')
	Data.q_table = np.zeros([len(data), len(data[0])])
	for j in range(len(data)):
		for k in range(len(data[j])):
			Data.q_table[j,k] = float(data[j][k])

	info = info.split('-')
	f.close()
		
	return int(info[0]), int(info[1]), int(info[2])

def saveQTableToFile(num_row, num_col, num_agents, filename):
	f = open(filename, "w")
	data = ""
	data = ';'.join([','.join([str(x) for x in row]) for row in Data.q_table])
	data = data+"?"+str(num_row)+'-'+str(num_col)+'-'+str(num_agents)
	f.write(data)
	f.close()

def extractActionFromSubState(state, agent_id):
	agents_row, agents_col = Data.env.decode(state)
	options = []

	for k in range(Data.env.num_agents):
		if Data.env.isAgentDead[k]==False and k!=agent_id:
			temp_state = Data.env.encode([agents_row[agent_id],agents_row[k]], [agents_col[agent_id],agents_col[k]])
			temp_action = Policy.bestFirstPolicy(temp_state)
			temp_value = Data.q_table[temp_state, temp_action]
			options.append([temp_action, temp_value])

	action = options[0][1]
	value = options[0][0]
	for i in range(len(options)):
		if options[i][0] > value:
			value = options[i][0]
			action = options[i][1]
	return action