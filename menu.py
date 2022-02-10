def menu():
	num_agents = 2
	num_dimensions = 2
	simulation_name = "listThreeKillers"
	use_existing = True
	save_results = False

	print("This project is made to train agents to fight each other\nThere is three types of agents\n-dummy : don't do anything\n-runner : just moving\n-killer : move and shoot\nWe are only using dummies and runners for the three first basic levels\n\nYou will now choose the parameters of the game\n")
	skipParam = input("skip and use default setup ? (best and latest trained agents) Y/N\n")
	if skipParam == "Y" or skipParam == "y":
		pass
	elif skipParam == "N" or skipParam == "n":
		answer = input("Select the number corresponding to the simulation you want to make:\n1: killer vs dummy (only killer is an agent, 2D)\n2: killer vs runner (only killer is an agent, 2D)\n3: killer vs runner (both are agents, 2D)\n4: killer vs killer (2D)\n5: three killers (2D)\n")
		if answer=='1':
			num_dimensions = 2
			num_agents = 2
			simulation_name = "killerVsDummy"
		elif answer=='2':
			num_dimensions = 2
			num_agents = 2
			simulation_name = "killerVsRunner"
		elif answer=='3':
			num_dimensions = 2
			num_agents = 2
			simulation_name = "listKillerVsRunner"
		elif answer=='4':
			num_dimensions = 2
			num_agents = 2
			simulation_name = "listKillerVsKiller"
		elif answer=='5':
			num_dimensions = 2
			num_agents = 2
			simulation_name = "listThreeKillers"
		else:
			print("wrong value selected")


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

	print("\nYou have selected : "+str(simulation_name)+", using trained agents:"+str(use_existing)+", saving results:"+str(save_results))
	return num_agents, num_dimensions, use_existing, save_results, simulation_name

	