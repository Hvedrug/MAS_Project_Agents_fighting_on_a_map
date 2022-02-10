import numpy as np
from gym import Env, spaces, utils
from random import randrange


MAP = [
"+---------+",
"| | | | | |",
"| | | | | |",
"| | | | | |",
"| | | | | |",
"| | | | | |",
"+---------+",
]


class Environment:
    """
    inspired by Tom Dietterich work on the Taxi Problem, Taxi-v3 from gym Env library

    Description:
    This work has been made for my multi-agent system class at the University of Genova.
    The subject was the training of two agents to gun fight against each other by using Q-learning algorithms.
    The agents are on a grid (see MAP) and they can move in the four cardinal directions and shoot in the same directions.
    If agent1 fire toward agent2 he wins, and agent2 looses. 

    various approaches of implementation:
    - tpt actions (agent1, agent2, agent1, agent2, ...) 
    - synchronous actions, do move before shoot. Having agent1 using only random actions and training agent2 to win.
    - synchronous actions, move before shoot. both agents training with q-learning.
    - more than 2 agents, same rules as before.
    - asynchronous actions, agent1 random actions
    - asynchronous actions, both on q-learning
    - change map size to a bigger one, try thinking for 3D and changing size agents (kneeling, jumping, ...)

    Can re-use already trained agents by using a Q-learning table and storing it in a txt file to import it later instead of using np.zeros()

    agent arguments :
    Name
    Type (to know which step() function to use) or maybe we can do one file per agent type 
    q-learning table location if requiered ('' => np.zeros)



    MAP:
        +---------+
        | | | | | |
        | | | | | |
        | | | | | |
        | | | | | |
        | | | | | |
        +---------+

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: fire south
    - 5: fire north
    - 6: fire east
    - 7: fire west

    Observations:
    There are 600 discrete states since there are 25 positions for each of the two agents.

    Rewards:
    - -1 per step reward unless other reward is triggered.
    - +20 killing oponent.
    - -2 if wrong fire. 
    - -10 wrong move (out of the map)
    - -20 get killed


    ```
    gym.make('Taxi-v3')
    ```

    """

    metadata = {'render.modes': ['human']}

    """
    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")
        num_states = 600
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 8
    """

    def __init__(self):
        super(Environment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(8)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.desc = np.asarray(MAP, dtype="c")

        self.num_agents = 2
        self.lastActions = [0 for _ in range(self.num_agents)]
        self.isAgentDead = [False for _ in range(self.num_agents)]
        self.num_rows = 5
        self.num_columns = 5
        self.num_states = (self.num_rows*self.num_columns)**self.num_agents
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = 8
        self.P = {
            state: {
            action: { 
            agent: [] 
            for agent in range(self.num_agents)} 
            for action in range(self.num_actions)}
            for state in range(self.num_states)
        }

        
        for state in range(self.num_states):
            for agent in range(self.num_agents):
                for action in range(self.num_actions):

                    ags_row, ags_col = self.decode(state)
                    n_ags_row = ags_row.copy()
                    n_ags_col = ags_col.copy()
                    # agent location: ags_row[agent], ags_col[agent]
                    reward = -1
                    done = False
                        
                    if action == 0:
                        n_ags_row[agent] = min(ags_row[agent] + 1, self.max_row)
                        if ags_row[agent] == self.max_row:
                            reward = -10
                    elif action == 1:
                        n_ags_row[agent] = max(ags_row[agent] - 1, 0)
                        if ags_row[agent] == 0:
                            reward = -10
                    elif action == 2:
                        n_ags_col[agent] = min(ags_col[agent] + 1, self.max_col)
                        if ags_col[agent] == self.max_col:
                            reward = -10
                    elif action == 3:
                        n_ags_col[agent] = max(ags_col[agent] - 1, 0)
                        if ags_col[agent] == 0:
                            reward = -10

                    elif action == 4:
                        for opponent in range(self.num_agents):
                            if opponent!=agent:
                                if n_ags_col[agent]==n_ags_col[opponent] and n_ags_row[agent]<=n_ags_row[opponent]:
                                    done = True
                                    reward = 20
                        if reward == -1:
                            reward = -2
                    elif action == 5: 
                        for opponent in range(self.num_agents):
                            if opponent!=agent:
                                if n_ags_col[agent]==n_ags_col[opponent] and n_ags_row[agent]>=n_ags_row[opponent]:
                                    done = True
                                    reward = 20
                        if reward == -1:
                            reward = -2
                    elif action == 6:
                        for opponent in range(self.num_agents):
                            if opponent!=agent:
                                if n_ags_col[agent]<=n_ags_col[opponent] and n_ags_row[agent]==n_ags_row[opponent]:
                                    done = True
                                    reward = 20
                            if reward == -1:
                                reward = -2
                    elif action == 7: 
                        for opponent in range(self.num_agents):
                            if opponent!=agent:
                                if n_ags_col[agent]>=n_ags_col[opponent] and n_ags_row[agent]==n_ags_row[opponent]:
                                    done = True
                                    reward = 20
                        if reward == -1:
                            reward = -2
                    new_state = self.encode(
                        n_ags_row, n_ags_col
                    )
                    self.P[state][action][agent].append((1.0, new_state, reward, done))

        #self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)


    def encode(self, agents_row, agents_col):
        # how to go from locations of agents to state value 
        # return encoded_data
        # agents_row et agents_col are arrays of int of size self.num_agents
        res = 0
        if (len(agents_row)!=self.num_agents or len(agents_col)!=self.num_agents):
            res = -1
        else:
            for i in range(self.num_agents):
                res += agents_row[i]
                res *= self.num_rows
                res += agents_col[i]
                res *= self.num_columns
            res = res/self.num_columns
        return res

    def decode(self, i):
        # how to go from state value to locations of agents 
        # return decoded_data
        agents_row = []
        agents_col = []
        for _ in range(self.num_agents):
            agents_col.append(i % self.num_rows)
            i = i // self.num_rows
            agents_row.append(i % self.num_columns)
            i = i // self.num_columns
        i*=self.num_columns
        assert 0 <= i < self.num_columns, "env.decode translation error"
        assert len(agents_col)==self.num_agents, "env.decode error len(agents_col)"
        assert len(agents_row)==self.num_agents, "env.decode error len(agents_row)"
        agents_row = list(reversed(agents_row))
        agents_col = list(reversed(agents_col))
        return agents_row, agents_col

    def step(self, actions):
        # actions is the list of the actions to perform
        # return (new state, rewards, dones, infos)
        # for now it is agent1 then agent2 then ...
        new_state = self.s
        rewards = []
        dones = []
        infos = []
        end = False
        # shoot first
        for i in range(len(actions)):
            if actions[i]>=4:
                transitions = self.P[self.s][actions[i]][i]
                p, s, r, d = transitions[0]
                new_state = s
                self.s = new_state
                rewards.append(r)
                if r==20:
                    self.isAgentDead[(i+1)%self.num_agents] = True
                dones.append(d)
                if d == True:
                    end = d
                infos.append({"prob": p})
        # move then
        if not end:
            for i in range(len(actions)):
                if actions[i]<4:
                    transitions = self.P[self.s][actions[i]][i]
                    p, s, r, d = transitions[0]
                    new_state = s
                    self.s = new_state
                    rewards.append(r)
                    dones.append(d)
                    infos.append({"prob": p})
        else:
            for i in range(len(actions)):
                if actions[i]<4:
                    p, s, r, d = "", self.s, -20, True
                    new_state = s
                    rewards.append(r)
                    dones.append(d)
                    infos.append({"prob": p})
        for i in range(len(actions)):
            if self.isAgentDead[i] == True:
                rewards[i] = -20
        self.s = new_state
        self.lastActions = actions
        return (int(self.s), rewards, dones, infos)

    def reset(self):
        # return int(self.s)
        self.s = randrange(self.num_states)
        return self.s

    def render(self, dones, ags_row, ags_col, actions):
        table = [["" for _ in range(self.num_columns)] for _ in range(self.num_rows)]
        for i in range(len(ags_col)):
            table[ags_row[i]][ags_col[i]] += str(i) 
        for j in range(len(actions)):
            if actions[j] == 4: #if agent fire south
                for i in range(self.num_rows):
                    if i > ags_row[j]:
                        table[i][ags_col[j]] += "."
            if actions[j] == 5: #if agent 1 fire north
                for i in range(self.num_rows):
                    if i < ags_row[j]:
                        table[i][ags_col[j]] += "."
            if actions[j] == 6: #if agent 1 fire east
                for i in range(self.num_columns):
                    if i > ags_col[j]:
                        table[ags_row[j]][i] += "."
            if actions[j] == 7: #if agent 1 fire west 
                for i in range(self.num_columns):
                    if i < ags_col[j]:
                        table[ags_row[j]][i] += "."
            if self.isAgentDead[j]: #if agent is dead
                table[ags_row[j]][ags_col[j]] = (table[ags_row[j]][ags_col[j]]).replace(str(j), "X")
        result = "+---------+"
        for i in range(len(table)):
            result+="\n|"
            for j in range(len(table[i])):
                if table[i][j]=="":
                    table[i][j]=" "
                result+=str(table[i][j])+"|"
        result += "\n+---------+\n\n\n"
        print(result)
