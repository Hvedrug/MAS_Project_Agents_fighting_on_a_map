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

        self.num_states = 625
        self.num_rows = 5
        self.num_columns = 5
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = 8
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }

        # hard coded policy
        for row_ag1 in range(self.num_rows):
            for col_ag1 in range(self.num_columns):
                for row_ag2 in range(self.num_rows):
                    for col_ag2 in range(self.num_columns):
                        
                        state = self.encode(row_ag1, col_ag1, row_ag2, col_ag2)
                        
                        for action in range(self.num_actions):
                            # defaults
                            new_row_ag1, new_col_ag1, new_row_ag2, new_col_ag2 = row_ag1, col_ag1, row_ag2, col_ag2
                            reward = -1  # default reward when there is no pickup/dropoff
                            done = False
                            ag1_localisation = (row_ag1, col_ag1)
                            ag2_localisation = (row_ag2, col_ag2)

                            if action == 0:
                                new_row_ag1 = min(row_ag1 + 1, self.max_row)
                                if row_ag1 == self.max_row:
                                    reward = -10
                            elif action == 1:
                                new_row_ag1 = max(row_ag1 - 1, 0)
                                if row_ag1 == 0:
                                    reward = -10
                            elif action == 2:
                                new_col_ag1 = min(col_ag1 + 1, self.max_col)
                                if col_ag1 == self.max_col:
                                    reward = -10
                            elif action == 3:
                                new_col_ag1 = max(col_ag1 - 1, 0)
                                if col_ag1 == 0:
                                    reward = -10
                            elif action == 4:  
                                if new_col_ag1==new_col_ag2 and new_row_ag1<=new_row_ag2:
                                    done = True
                                    reward = 20
                                else:
                                    reward = -2
                            elif action == 5: 
                                if new_col_ag1==new_col_ag2 and new_row_ag1>=new_row_ag2:
                                    done = True
                                    reward = 20
                                else:
                                    reward = -2
                            elif action == 6:
                                if new_col_ag1<=new_col_ag2 and new_row_ag1==new_row_ag2:
                                    done = True
                                    reward = 20
                                else:
                                    reward = -2
                            elif action == 7: 
                                if new_col_ag1>=new_col_ag2 and new_row_ag1==new_row_ag2:
                                    done = True
                                    reward = 20
                                else:
                                    reward = -2
                            new_state = self.encode(
                                new_row_ag1, new_col_ag1, new_row_ag2, new_col_ag2
                            )
                            self.P[state][action].append((1.0, new_state, reward, done))

        #self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)




    def encode(self, agent1_row, agent1_col, agent2_row, agent2_col):
        # how to go from locations of agents to state value 
        # return encoded_data
        i = agent1_row
        i *= self.num_rows
        i += agent1_col
        i *= self.num_columns
        i += agent2_row
        i *= self.num_rows
        i += agent2_col
        return i

    def decode(self, i):
        # how to go from state value to locations of agents 
        # return decoded_data
        # value in [0, 625]
        out = []
        out.append(i % self.num_rows)
        i = i // self.num_rows
        out.append(i % self.num_columns)
        i = i // self.num_columns
        out.append(i % self.num_rows)
        i = i // self.num_rows
        out.append(i)
        assert 0 <= i < 5
        a1r, a1c, a2r, a2c = reversed(out)
        return a1r, a1c, a2r, a2c

    def step(self, action):
        # ...
        # return (new state, reward, done, info)
        transitions = self.P[self.s][action]
        # print(transitions)
        # choose policy
        # i = random_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[0]
        self.s = s
        self.lastaction = action
        return (int(s), r, d, {"prob": p})

    def reset(self):
        # return int(self.s)
        self.s = randrange(self.num_states)
        return self.s

    def render(self, done, agent1_row, agent1_col, agent2_row, agent2_col, action=-1):
        table = [[0 for _ in range(self.num_columns)] for _ in range(self.num_rows)]
        table[agent1_row][agent1_col] = 1 
        if action == 4: #if agent 1 fire south
            for i in range(self.num_rows):
                if i > agent1_row:
                    table[i][agent1_col] = 3
        if action == 5: #if agent 1 fire north
            for i in range(self.num_rows):
                if i < agent1_row:
                    table[i][agent1_col] = 3
        if action == 6: #if agent 1 fire east
            for i in range(self.num_columns):
                if i > agent1_col:
                    table[agent1_row][i] = 3
        if action == 7: #if agent 1 fire west 
            for i in range(self.num_columns):
                if i < agent1_col:
                    table[agent1_row][i] = 3
        if done: #if agent 2 is dead
            table[agent2_row][agent2_col] = 8
        else:
            table[agent2_row][agent2_col] = 2
        result = "+---------+"
        for i in range(len(table)):
            result+="\n|"
            for j in range(len(table[i])):
                if (table[i][j]==0):
                    result+=" |"
                elif (table[i][j]==3):
                    result+=".|"
                elif (table[i][j]==8):
                    result+="X|"
                else:
                    result+=str(table[i][j])+"|"
        result += "\n+---------+\n\n\n"
        print(result)
