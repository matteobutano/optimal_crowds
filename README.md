# Optimal Crowds :crystal_ball:

The optimal_crowds package simulates the dynamics of a crowd of human beings moving through an environment towards a target. Once the room is set up, an HJB equation analogous to that found in [Bonnemain et al.](https://arxiv.org/pdf/2201.08592) is solved to find the optimal trajectories leading to the targets. The motion of pedestrians is then simulated using a Agent-Based Model where agents strive the follow the optimally chosen trajectories and avoid obstacles and others via an adaptation of the Generalized Centrifugal-Force Model (GCFM) detailed in [Chraibi et al.](https://arxiv.org/pdf/1008.4297). 

### INSTALL 💻

To correctly set up your package: 
- Have the python modules numpy, matplotlib, json and scipy installed
- Clone the repository inside a directory, you can do that by using the command 'git clone https://github.com/matteobutano/optimal_crowds' in you favorite bash terminal 
- Inside the directory where you cloned the repo, **not inside the repo itself**, create a folder named 'rooms', where you will place your room configurations 

### CREATE YOUR ROOM 🔨

In the 'rooms' folder, you will place the .json files containing all the information about the configurations you wish to simulate. You can move the 'room_test.json' file from this repo to your 'rooms' folder and modify it. The main elements of the room configuration file are:
- **room_legth** and **room_height**: the extension along the x and y axes of the simulation room
- **initial_boxes**: telling the rectagular regions of the simulation room we initialize the agents. Each box is an array telling in order: the rectangle's center's x coordinate, y coordinate, horizontal width, vertical width, the value of the average density in the initial box, measured in ped/m² and all the targets of agents spawned in that box. Agents will reach one of the listed target depending on the strategy given by the HJB solution
- **targets**: the target areas of pedestrians' motion. Each door is an array telling in order: the door's center's x coordinate, y coordinate, horizontal width, vertical width
- **walls**: rectangular obstacles placed in the simulation room. Each wall is an array telling in order: the wall's center's x coordinate, y coordinate, horizontal width, vertical width 
- **holes**: openings in walls. Each hole is an array telling in order: the the hole's center's x coordinate, y coordinate, horizontal width, vertical width 
- **cylinders**: cylindrical obstacles in the simulation room. Each cylinder is an array telling in order: the cylinder's center's x coordinate, y coordinate, the cylinder radius

### START YOUR FIRST SIMULATION ▶️

In the directory where you cloned the 'optimal_control' repository, create a python script 'run.py' with instructions:
1. **'from optimal_crowds import simulations'**, to import the simulation module 
2. **'simu = simulations.simulation(room, T)'**, to create the simulation room , where: *room*, must be a string with the name without extension of the room's configuration file saved in you 'rooms' folder; *T* must be a float determining the max time in seconds you allow agents to exit the simulation room. If T is too small, agents won't move from their initial positions; in that case, try increasing T
3. **'simu.draw()'**, to visualize the initial configuration 
4. **'simu.run(draw, verbose)'**, to execute the simulation , where: *draw* must be boolean. If True the simulation room and the agents are plotted at each time step; *verbose* must be a boolean. If True the simulation time in seconds and the number of exited agents are printed at each time step
5. **'simu.draw_final_trajectories()'** to finally, plot the actual trajectory each agent followed to exit the simulation room using 

### CONTRIBUTE 🏁

If you like what this does, feel free to improve upon code. Just follow these steps to contribute:

1. Fork it
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Issue a pull request
