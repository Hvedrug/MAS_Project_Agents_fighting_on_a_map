import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import menu as mn


def launchEnv(use_existing, save_results):
	if use_existing:
		launch.TestExistingAgent()
	elif save_results:
		launch.TrainingSaveAndTestOne()
	else:
		launch.TrainingAndTestOne()


use_existing, save_results, simulation_name = mn.menu()

if simulation_name == "killerVsDummy":
	import src.killerVsDummy.Launch as launch
elif simulation_name == "killerVsRunner":
	import src.killerVsRunner.Launch as launch
elif simulation_name == "listKillerVsRunner":
	import src.listKillerVsRunner.Launch as launch
elif simulation_name == "listKillerVsKiller":
	import src.listKillerVsKiller.Launch as launch
elif simulation_name == "listThreeKillers":
	import src.listThreeKillers.Launch as launch
elif simulation_name == "listWithOptions":
	import src.listWithOptions.Launch as launch
elif simulation_name == "listWithOptionsOptimized":
	import src.listWithOptionsOptimized.Launch as launch
else:
	import src.listWithOptionsOptimized.Launch as launch

launchEnv(use_existing, save_results)
