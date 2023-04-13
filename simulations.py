# author Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

# Modules are imported 

import numpy as np
import matplotlib.pyplot as plt
from optimal_crowds import pedestrians
from optimal_crowds import optimals
import json

# The simulation class creates the simulation room and makes it evolve 
# accordingly to the mode indicated as argument. The 'abm' creates an agent based model 
# simulation where agents strive to reach the exit while avoiding obstacle, always being 
# guided by the optimal velocities calculated by solving the HJB equation. 

class simulation:
    
    def __init__(self,room, mode,T = 10):
        
        # The config.json contains the parameters of the abm and of the mfg system 
        
        self.type = mode
        
        with open('optimal_crowds/config.json') as f:
            var_config = json.loads(f.read())
        
        # The door.json file contains the description of the simulation room. 
        
        with open('rooms/'+room+'.json') as f:
            var_room = json.loads(f.read())
        
        # We first create the simulation room, by determining its length,
        # i.e. its extension on the x-axis, its height, i.e. its extension
        # on the y-axis 
        
        self.room_length = var_room['room_length']
        self.room_height = var_room['room_height']
        
        # A grid is defined that will serve as the space discretization for the 
        # MFG equations and the calculation of the density generated by gaussian 
        # convolution of the agents positions in the abm. 
        
        self.grid_step = var_config['grid_step']
        
        self.Nx = int(self.room_length//self.grid_step + 1)
        self.Ny = int(self.room_height//self.grid_step + 1)
        
        self.dx = self.grid_step
        self.dy = self.grid_step

        self.X_opt, self.Y_opt = np.meshgrid(np.linspace(0,self.room_length,self.Nx)
                                             ,np.linspace(0,self.room_height,self.Ny))
        
        self.sigma_convolution = var_config['sigma_convolution']
        
        # Here we initialise the terminal condition of the Cole-Hopf transformed
        # value function, from which, solving the HJB equation, 
        # we will obstain the optimal trajectories
        
        self.u_0 = np.zeros((self.Ny,self.Nx),dtype = float)
        self.phi_0 =  np.zeros((self.Ny,self.Nx),dtype = float)
        self.evacuator = np.zeros((self.Ny,self.Nx),dtype = float) + 1
        
        # Here the walls are created, and they will be used both 
        # as the potential representing the interactions with the 
        # environment in the MFG and the repulsive walls of the abm
        
        self.pot = var_config['hjb_params']['wall_potential']
        self.V = np.zeros((self.Ny,self.Nx)) + self.pot
        self.V[1:-1,1:-1] = 0
        self.lim = 10e-6
        
        for walls in var_room['walls']:
            wall = var_room['walls'][walls]

            mask_X = abs(self.X_opt-wall[0]) < wall[2]/2
            mask_Y = abs(self.Y_opt-wall[1]) < wall[3]/2          
            
            V_temp = np.zeros((self.Ny,self.Nx))
            V_temp[mask_X*mask_Y] = self.pot
            
            self.V += V_temp
         
        for holes in var_room['holes']:
            hole = var_room['holes'][holes]
        
            hole_X = abs(self.X_opt-hole[0]) < hole[2]/2
            hole_Y = abs(self.Y_opt-hole[1]) < hole[3]/2
            
            self.V[hole_X*hole_Y] = 0   
        
        for cyls in var_room['cylinders']:
            cyl = var_room['cylinders'][cyls]
            
            V_temp =  np.zeros((self.Ny,self.Nx))
            V_temp[np.sqrt((self.X_opt-cyl[0])**2 + (self.Y_opt-cyl[1])**2) < cyl[2]] = self.pot
            
            self.V+= V_temp
        
        self.V = self.pot *(self.V <= self.pot)  
        
        self.doors = np.empty((len(var_room['doors']),4))
        for i,door in enumerate(var_room['doors']):
            self.doors[i,:] = np.array(var_room['doors'][str(door)])
        
        for door in var_room['doors']:
            door = var_room['doors'][door]
            
            door_X = abs(self.X_opt - door[0]) < door[2]/2
            door_Y = abs(self.Y_opt - door[1]) < door[3]/2
           
            self.evacuator[door_X*door_Y] = 0
            self.V[door_X*door_Y] = var_config['hjb_params']['door_potential']
            
        self.phi_0 = self.phi_0.reshape(self.Nx*self.Ny)
        
        # Here the time discretization is defined
        
        self.T = T
        self.time = 0.
        self.simu_step = 0
        self.dt = var_config['dt']
        
        # Here the main parameters of the social force model regulating the
        # interaction among agents are read and initialized
        
        self.relaxation = var_config['relaxation']
        self.rep_radius = var_config['repulsion_radius']
        self.rep_int = var_config['repulsion_intensity']
        self.noise_intensity = var_config['hjb_params']['sigma']
        self.des_v = var_config['des_v']
        
        # We create the object containing the optimal trajectories for the abm 
        # and also the one able to compute the mfg
        
        self.optimal = optimals.optimals(room,T)
        
        # Finally, depending on the mode of the simulation, we initialize the crowd 
        # both for the abm and the mfg
        
        N = 0
        self.agents =[]
        self.m_0 = np.zeros((self.Ny,self.Nx),dtype = float)
        
        if self.type == 'abm':
            for boxes in var_room['initial_boxes']:
                
                box = var_room['initial_boxes'][boxes]
                
                loc_N = int(box[4]*(box[1]-box[0])* (box[3]-box[2]))
                
                N += loc_N
               
                xs = np.random.uniform(box[0] - box[2]/2,box[0] + box[2]/2,loc_N)
                ys = np.random.uniform(box[1] - box[3]/2,box[1] + box[3]/2,loc_N)
                
                for i in range(loc_N):
                    
                    # Here we create every single agent using the pedestrian module, 
                    # each agent is therefore and instance of the object pedestrian,
                    # which allows us to monitor the various parameters of each agent's
                    # dynamics, such as speed, position, direction, target, evacuation time etc.
                   
                    self.agents.append(pedestrians.ped(xs[i], ys[i], 0, 0, self.doors, self.room_length, self.room_height))
            
            self.N = N
            self.inside = self.N
            self.agents = np.array(self.agents, dtype = object)
               
            print('ABM simulation room created!')
                     
            self.optimal.compute_optimal_velocity()
        
        elif self.type =='mfg':
            
            for boxes in var_room['initial_boxes']:
                
               box = var_room['initial_boxes'][boxes]
            
               # Here the mfg density is created  
                
               x_min = box[0] - box[2]/2
               x_max = box[0] + box[2]/2
               y_min = box[1] - box[3]/2
               y_max = box[1] + box[3]/2
               dens = box[4]
               
               X = self.X_opt
               Y = self.Y_opt
               self.m_0[((X > x_min) & (X < x_max)) * ((Y > y_min) & (Y < y_max))] = dens
        
            print('MFG simulation room created!')
        
    # The 'draw' method allows for visualisation of the simulation room
    
    def draw(self,mode = 'scatter'):
        
        plt.figure(figsize = (self.room_length,self.room_height))
        
        if self.type == 'abm':
            
            # In abm mode, this method draws a snapshot of the simulation room at each time step
        
            if mode == 'scatter':
                
            # Where each agent is represented by a dot of radius 0.2m
            
                for i in range(self.N): 
                    if self.agents[i].status:
                        c = self.agents[i].position()
                        C = plt.Circle(c,radius = 0.2)
                        plt.gca().add_artist(C)
                plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
                plt.xlim([0,self.room_length])
                plt.ylim([0,self.room_height])
                title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
                plt.title(title)
                plt.show()
             
            if mode == 'arrows':
                
            # Where each agent is represented by an arrow indicating its velocity 
            
                if self.inside > 0:
                    scat_x = [self.agents[i].position()[0] for i in range(self.N) if self.agents[i].status]
                    scat_y = [self.agents[i].position()[1] for i in range(self.N) if self.agents[i].status]
                    scat_vx = [self.agents[i].velocity()[0] for i in range(self.N) if self.agents[i].status]
                    scat_vy = [self.agents[i].velocity()[1] for i in range(self.N) if self.agents[i].status]
                    plt.quiver(scat_x,scat_y,scat_vx, scat_vy,color = 'blue')
                else:
                    plt.plot() 
                plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
                plt.xlim([0,self.room_length])
                plt.ylim([0,self.room_height])
                title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
                plt.title(title)
                plt.show()
            
            if mode == 'density':
                
            # Where the gaussian convolution of the agents position is computed  
                
                d = self.gaussian_density(self.sigma_convolution, self.Nx, self.Ny)
                plt.imshow(np.flip(self.V/self.pot,axis = 0) + d,extent=[0,self.room_length,0,self.room_height])
                plt.xlim([0,self.room_length])
                plt.ylim([0,self.room_height])
                plt.colorbar()
                plt.clim(0,2)
                title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
                plt.title(title)
                plt.show()
            
        elif self.type == 'mfg':
            
            plt.pcolor(self.X_opt, self.Y_opt,self.m_0 + self.V/self.pot)
                    
    # The 'step' method puts together all the ingredients necessary to 
    # evolve each agent's position and status in the simulation
            
    def step(self,dt,verbose =  False):
        
        # First, we summon each agent randomly
        
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            
            agent = self.agents[i]
    
            # Check if agent has reached the a door
            
            if agent.status:
            
                # We assign to the agent the velocity prescribed optimally by the HJB equation
                
                des_x, des_y  = self.optimal.choose_optimal_velocity(agent.position(), self.simu_step)
                
                # We compute the repulsion from other agents
                
                repulsion = np.array((0, 0),dtype = float)
                for j in range(self.N):
                    if self.agents[j].status and j != i:
                        repulsion = repulsion + np.array(agent.compute_repulsion(self.agents[j].position(), self.rep_radius, self.rep_int),dtype = float)
                        
                # We compute repulsion from walls
                
                wall_repulsion = np.array(agent.wall_repulsion(self.rep_radius, self.rep_int,self.X_opt,self.Y_opt,self.V),dtype=float)
                
                # We compute current velocity with random perturbation and repulsions
                
                current_velocity = agent.velocity() + 0.5*self.noise_intensity*np.random.normal(size = 2) + repulsion + wall_repulsion 
        
                # We compute acceleration towards desired velocity
                
                ax = (des_x - current_velocity[0]) / self.relaxation
                ay = (des_y - current_velocity[1]) / self.relaxation
            
                # We compute new position and velocity
                
                old_x = agent.position()[0]
                old_y = agent.position()[1]
                x = old_x + current_velocity[0] * dt + 0.5 * ax * dt**2
                y = old_y + current_velocity[1] * dt + 0.5 * ay * dt**2
                vx = current_velocity[0] + ax * dt
                vy = current_velocity[1] + ay * dt
                
                # We update agent position
                
                agent.evolve(x, y, vx, vy, dt)
                
                # We finally check if agent has left the room
                
                agent.check_status()
                if agent.status == False:
                    self.inside +=-1
                    
                # And choose new target
                
                agent.choose_target()
        
        # After all agents have been summoned and their status updated, 
        # we advance time and simu step, to keep track of the time evolution of the simulation
        
        self.time+=dt
        self.simu_step+=1
        
        if verbose:
            print('t = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N),end='\n')
            
    # The method evac_times allows to obtain the evacuation time, 
    # i.e., the time needed to exit the simulation room, for each agent.
    # If the option draw is on, each agent's starting position is plotted as a dot
    # whose color represents the time needed to evacuate the room        
                
    def evac_times(self,draw = False):
        if self.inside > 0:
            raise ValueError('There are still people inside!')
        times = np.empty(self.N,dtype = float)
        for i, agent in enumerate(self.agents):
            times[i] = agent.time
        if draw: 
            xs,ys = self.initial_positions()
            plt.scatter(xs, ys, c=times)
            plt.xlim(0,self.room_length)
            plt.ylim(0,self.room_height)
            title = 'Evacuation time'
            plt.title(title)
            plt.colorbar()
        else:
            return times
        
    # The 'initial_position' method extracts the initial position of each agent
    # of the simulation and provides a list with all tuples 
            
    def initial_positions(self):
        xs = np.empty(self.N,dtype = float)
        ys = np.empty(self.N,dtype = float)
        for i, agent in enumerate(self.agents):
            xs[i] = agent.initial_position[0]
            ys[i] = agent.initial_position[1]
        return xs,ys

    # The 'run' methods makes the simulation evolve depending on the selected mode. 
    # In 'abm' mode, it applies a step of given time-step until all agents have exited 
    # the room or until the simulation time is lower than the prescribed one. This 
    # is because the optimal velocities are only computed up to the terminal time T.
    
    def run(self, verbose = True, draw = False, mode = 'scatter'):
        
        # In 'abm', the simulation is updated one step at the time, 
        # following the rules of the 'step' method
        
        if self.type == 'abm':
            
            while (self.inside > 0) & (self.time < self.T): 
                
                self.step(self.dt,verbose = verbose)
                
                # Draw current state of the simulation 
                
                if draw:
                    self.draw(mode)
                    plt.show()
                    
            # Print evacuation status
            
            if self.inside == 0:
                print('Evacuation complete!')
            else:
                print('Evacuation failed!')
          
        # In 'mfg' mode a the Nash equillibrium of the system is reached
        # by cycling over the solutions of the Cole-Hopf version of HJB and
        # KFP equations. 
        
        elif self.type == 'mfg':
            
            self.optimal.mean_field_game(self.m_0,draw = draw,verbose=verbose)
       
    # The 'gaussian_density' method computes the gaussian convolution of the agents
    # positions. Each position is the center of a gaussian with standard deviation
    # given by the 'sigma_convolution' paramter.      
    
    def gaussian_density(self,sigma,Nx,Ny):
        
        dx = self.grid_step
        dy = self.grid_step

        X,Y = np.meshgrid(np.linspace(0,self.room_length,self.Nx), 
                          np.linspace(0,self.room_height,self.Ny))
        
        X = X[:-1,:-1] + dx/2
        Y = np.flip(Y[:-1,:-1] + dy/2,axis = 0)
        d = np.zeros((Ny,Nx))
        count = 0
        
        for agent in self.agents:
            if agent.status:
                count+=1
                x_agent = agent.position()[0]
                y_agent = agent.position()[1]
                c_x = X - x_agent
                c_y = Y - y_agent
                C = np.sqrt(4*np.pi**2*sigma**2)
                d += np.exp(-(c_x**2 + c_y**2)/(2*sigma**2))/C 
                
        return d
    
    def draw_final_trajectories(self):
        
        plt.figure(figsize = (self.room_length,self.room_height))
        
        for i in range(self.N):
            self.agents[i].draw_trajectory()
        
        plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])
        
        plt.show()
        