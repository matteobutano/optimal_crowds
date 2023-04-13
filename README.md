# Optimal Crowds 

The optimal_crowds package helps you simulate the dynamics of a crowd of human beings moving through an environment towards an exit or a goal. The algorithms allows for  two methods of simulation, an Agent-Based Model (ABM) and a Mean-Field Game (MFG). Once the room has been set up, both methods can be chosen to perform the simulation. 

### How to install

To correctly set up your simulation: 
- clone the repository inside a directory, you can do that by using the command 'git clone https://github.com/matteobutano/optimal_crowds' in you favorite bash terminal. 
- inside the same directory create a folder named 'rooms', where you will place your room configurations 

### How to create a room 

In the 'rooms' folder, you will place the .json files containing all the information about the configurations you wish to simulate. For starters, you can move the 'room_test.json' file from this repo to your 'rooms' folder. The room files must contain the following:
- dt: the time step of the simulation
- room_length and room_height: the extension along the x and y axes of the simulation room
- Nx and Ny: the number of points of the space discretization along the x and y axes 
- initial_boxes: telling the rectagular regions of the simulation room we initialize the agents. Each box is an array telling in order: the rectangle's center's x coordinate, y coordinate, horizontal width, vertical width, the value of the average density in the initial box, mesured in ped/m²
- doors: the target areas of pedestrians' motion. Each door is an array telling in order: the door's center's x coordinate, y coordinate, horizontal width, vertical width, the door's importance. 
- walls: rectangular obstacles placed in the simulation room. Each wall is an array telling in order: the wall's center's x coordinate, y coordinate, horizontal width, vertical width 
- holes: openings in walls. Each hole is an array telling in order: the the hole's center's x coordinate, y coordinate, horizontal width, vertical width 
-cylinders: cylindrical obstacles in the simulation room. Each cylinder is an array telling in order: the cylinder's center's x coordinate, y coordinate, the cylinder radius

### How to run your first simulation 

In the directory where you cloned the 'optimal_control' repository, create a python script where you will:
- Import the simulation module 'from optimal_crowds import simulations'.
- Create the simulation room 'simu = simulations.simulation(room, mode, T)', where: *room*, must be a string with the name without extension of the room's configuration file you want to simulate that you saved in you 'rooms' folder; *mode* must be the string 'abm' if you want the ABM, or the string 'mfg' if you want the MFG; *T* must be a float determining the max time you allow agents to exit the simulation room.
- Visualize the initial configuration 'simu.draw()'.
- Execute the simulation 'simu.run(draw, verbose)', where: *draw* must be boolean that let you choose if you want to plot the simulation; *verbose* must be a boolean that let you choose wether to display information about the evacuation in the case of the ABM, or the convergence of the self consistency cycles for the MFG.
- Only for the ABM simulation, use 'simu.draw_final_trajectories()' to display the complete trajectory of each agent at the end of the evacuation. 

### Final notice

I hope this code can serve you helping you simulate your favourite configurations. However, please note that this is still early stage, beta version and sometimes strange behaviors of the agents could be observed. Usually, for the ABM when the allowed simulation time is too small, agents close to the door will try pass through the wall, because they really want to get to the exit, or, agents would stand still around their initial position because they predict they won't have enough time to exit the room. In those case, try augmentint T, the simulation final time. For the MFG, sometimes the self-consistence cycle can't converge, and either the error blows up or the early stopping criterion is reached. 

Please feel free to raise any issues, pull requests, signal bugs and configurations that the algorithms can't get right. 
