import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep

"""
This version exist only because it is possible that the OS will not like the way I dynamically import some source files
Do not use this code if the other one (masMain.py) is working

In order to launch a simulation you need to manually change the few lines under this text. 
Uncomment the line that import the right simulation
Comment all the other imports
Save and execute this code (python masAlternativeVersion.py) 
"""

#import src.killerVsDummy.Launch as launch
#import src.killerVsRunner.Launch as launch
#import src.listKillerVsRunner.Launch as launch
#import src.listKillerVsKiller.Launch as launch
import src.listThreeKillers.Launch as launch


def launchEnv(use_existing, save_results):
	if use_existing:
		launch.TestExistingAgent()
	elif save_results:
		launch.TrainingSaveAndTestOne()
	else:
		launch.TrainingAndTestOne()

def menu():
	use_existing = True
	save_results = False

	print("This project is made to train agents to fight each other\nThere is three types of agents\n-dummy : don't do anything\n-runner : just moving\n-killer : move and shoot\nWe are only using dummies and runners for the three first basic levels\n\nYou will now choose the parameters of the game\n")
	skipParam = input("skip and use default setup ? (best and latest trained agents) Y/N\n")
	if skipParam == "Y" or skipParam == "y":
		pass
	elif skipParam == "N" or skipParam == "n":

		answer = input("Do you want to use the already trained agents ? Y/N\n")
		if answer == "Y" or answer == "y":
			#use_existing = True
			pass
		elif answer == "N" or answer == "n":
			use_existing = False
			save = input("Do you want to save results after the training ? Y/N\n")
			if save == "Y" or save == "y":
				save_results = True
			elif save == "N" or save == "n":
				pass
			else:
				print("wrong value selected")
		else:
			print("wrong value selected")
	else:
		print("wrong value selected")

	print("\nYou have selected : using trained agents:"+str(use_existing)+", saving results:"+str(save_results))
	return use_existing, save_results

	


use_existing, save_results = menu()
launchEnv(use_existing, save_results)