# Optimal Crowds 

The optimal_crowds package helps you simulate the dynamics of a crowd of human beings moving through an environment towards an exit or a goal. The algorithm allows for  two methods of simulation, an Agent-Based Model (ABM) and a Mean-Field Game (MFG). To know more about the theory behind read the wiki! Once the room is set up, both methods can be chosen to perform the simulation.

### How to install

To correctly set up your package: 
- Clone the repository inside a directory, you can do that by using the command 'git clone https://github.com/matteobutano/optimal_crowds' in you favorite bash terminal 
- Inside the same directory create a folder named 'rooms', where you will place your room configurations 

### How to create a room 

In the 'rooms' folder, you will place the .json files containing all the information about the configurations you wish to simulate. You can move the 'room_test.json' file from this repo to your 'rooms' folder and modify it. The main elements of the room configuration file are:
- room_length and room_height: the extension along the x and y axes of the simulation room
- initial_boxes: telling the rectagular regions of the simulation room we initialize the agents. Each box is an array telling in order: the rectangle's center's x coordinate, y coordinate, horizontal width, vertical width, the value of the average density in the initial box, measured in ped/m²
- doors: the target areas of pedestrians' motion. Each door is an array telling in order: the door's center's x coordinate, y coordinate, horizontal width, vertical width
- walls: rectangular obstacles placed in the simulation room. Each wall is an array telling in order: the wall's center's x coordinate, y coordinate, horizontal width, vertical width 
- holes: openings in walls. Each hole is an array telling in order: the the hole's center's x coordinate, y coordinate, horizontal width, vertical width 
-cylinders: cylindrical obstacles in the simulation room. Each cylinder is an array telling in order: the cylinder's center's x coordinate, y coordinate, the cylinder radius

### How to run your first simulation 

In the directory where you cloned the 'optimal_control' repository, create a python script 'run.py' where you will:
- Import the simulation module 'from optimal_crowds import simulations'.
- Create the simulation room 'simu = simulations.simulation(room, mode, T)', where: *room*, must be a string with the name without extension of the room's configuration file saved in you 'rooms' folder; *mode* must be the string 'abm' if you want the ABM, or the string 'mfg' if you want the MFG; *T* must be a float determining the max time you allow agents to exit the simulation room.
- Visualize the initial configuration 'simu.draw()'.
- Execute the simulation 'simu.run(draw, verbose)', where: *draw* must be boolean.If True the simulation is plotted at each time step; *verbose* must be a boolean. If True information about the ABM evacuation, or the convergence of the self consistency cycle for the MFG is displayed.
- Only for the ABM simulation, use 'simu.draw_final_trajectories()' to display the complete trajectory of each agent at the end of the evacuation. 

### Final notice

I hope this code can help you simulate your favourite configurations. However, please note that this is still an early stage, beta version and sometimes strange behavior of the agents could be observed. Usually, for the ABM when T is too small, agents may stand still around their initial position because they predict they won't have enough time to exit the room. In those case, try increasing T. For the MFG, sometimes the self-consistence cycle can't converge, and either the error blows up or the early stopping criterion is reached. 

Please feel free to raise any issues, pull requests and signal bugs, as this would greatly help the maintenance and improving of the code. 
