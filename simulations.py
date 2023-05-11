# author Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

# Modules are imported 

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.patches import Ellipse,Patch
from optimal_crowds import pedestrians
from optimal_crowds import optimals
import json
import seaborn as sns

# The simulation class creates the simulation room and makes it evolve 
# accordingly to the mode indicated as argument. The 'abm' creates an agent based model 
# simulation where agents strive to reach the exit while avoiding obstacle, always being 
# guided by the optimal velocities calculated by solving the HJB equation. 

class simulation:
    
    def __init__(self,room,T):
        
        # The config.json contains the parameters of the abm agents and
        # of the HJB equation used to guide their motion 
    
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
        self.pot = var_config['hjb_params']['wall_potential']
        self.lim = 10e-6
        
        # Here the time discretization is defined
        
        self.T = T
        self.time = 0.
        self.simu_step = 0
        self.dt = var_config['dt']
        
        # Here the main parameters of the social force model regulating the
        # interaction among agents are read and initialized
        
        self.relaxation = var_config['relaxation']
        self.noise_intensity = var_config['hjb_params']['sigma']
        self.a_min = var_config['b_min']
        self.tau_a = var_config['tau_a']
        self.b_min = var_config['b_min']
        self.b_max = var_config['b_max']
        self.eta = var_config['eta']
        self.repulsion_cutoff = var_config['repulsion_cutoff']
        
        
        # We initialize the crowd, we will count the number of agents and create 
        # the different potentials given by the combinations of targets
        
        N = 0
        self.agents =[]
       
        self.targets = {}
        self.Vs = {}
        self.V = np.zeros((self.Ny,self.Nx)) + 1
    
        for box in var_room['initial_boxes']:
                  
            box = var_room['initial_boxes'][box]
            
            targets = box[5:]
            key = ' or '.join(targets)
            
            V = self.create_potential(var_room,var_config,targets)
            self.Vs[key] = V
            self.targets[key] = optimals.optimals(room,V,T,key)
            self.V*= V
           
            loc_N = int(box[4] * box[2] * box[3])
            
            N += loc_N
              
            xs = np.random.uniform(box[0] - box[2]/2,box[0] + box[2]/2,loc_N)
            ys = np.random.uniform(box[1] - box[3]/2,box[1] + box[3]/2,loc_N)
            v_des_all = np.random.uniform(0.5,2,loc_N)
                
            for i in range(loc_N):
                    
                # Here we create every single agent using the pedestrian module, 
                # each agent is therefore and instance of the object pedestrian,
                # which allows us to monitor the various parameters of each agent's
                # dynamics, such as speed, position, direction, target, evacuation time etc.
                   
                self.agents.append(pedestrians.ped(self.Y_opt,self.X_opt,self.Vs[key],key, 
                                                   var_room['targets'],targets,
                                                   xs[i], ys[i], 0, 0, 
                                                   self.room_length, self.room_height,
                                                   v_des_all[i],self.a_min,self.tau_a,
                                                   self.b_min,self.b_max,self.eta))
            
        
        l = len(self.targets)  
        self.V = self.pot*(abs(self.V) == abs(self.pot)**l) + (abs(self.V) != abs(self.pot)**l)*(self.V !=0)
        
        self.N = N
        self.inside = self.N
        self.agents = np.array(self.agents, dtype = object)
               
        print('ABM simulation room created!')
                     
    # The 'draw' method allows for visualisation of the simulation room
    
    def draw(self,mode = 'scatter'):
        
        plt.figure(figsize = (self.room_length,self.room_height))
        colors = sns.color_palette(n_colors=len(self.targets))
        
        # This method draws a snapshot of the simulation room at each time step
    
        if mode == 'scatter':
            
        # Where each agent is represented by a dot of radius 0.2m
        
            for i in range(self.N): 
                if self.agents[i].status:
                    
                    agent = self.agents[i]
                    pos_i = agent.position()
                    vel_i = agent.velocity()
                    
                    direction = np.degrees(np.arctan2(vel_i[1],vel_i[0]))
            
                    a_i = agent.a_min + agent.tau_a * np.linalg.norm(vel_i)
                    b_i = agent.b_max - (agent.b_max - agent.b_min)*np.minimum(np.linalg.norm(vel_i)/agent.v_des,1)
                    
                    color_index = list(self.targets).index(agent.target)
                    
                    E = Ellipse(pos_i, width = a_i , height= b_i, angle = direction,
                                color = colors[color_index])
                    
                    plt.gca().add_artist(E)
                    
            plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
            plt.title(title)
            handles = [Patch(color=colors[i],label = list(self.targets)[i]) for i in range(len(self.targets))]
            plt.legend(handles = handles, loc = 'upper right', frameon = False)
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
            plt.imshow(np.flip(self.V/self.pot+ d,axis = 0) ,extent=[0,self.room_length,0,self.room_height])
            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            plt.colorbar()
            title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
            plt.title(title)
            plt.show()
        
                    
    # The 'step' method puts together all the ingredients necessary to 
    # evolve each agent's position and status in the simulation
            
    def step(self,dt,verbose =  False):
        
        # First, we summon each agent randomly
        
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            
            agent = self.agents[i]
    
            # Check if agent has reached the a door
            
            if agent.status:
            
                # We assign to the agent the velocity prescribed optimally by the HJB equation
                
                des_x, des_y  = agent.v_des*self.targets[agent.target].choose_optimal_velocity(agent.position(), self.simu_step)
                
                # We compute the repulsion from other agents
                
                repulsion = np.array((0, 0),dtype = float)
                for j in range(self.N):
                    pos_j = self.agents[j].position()
                    vel_j = self.agents[j].velocity()
                    if self.agents[j].status and j != i and agent.distance(pos_j) < self.repulsion_cutoff:
                        repulsion = repulsion + np.array(agent.agents_repulsion(pos_j,vel_j),dtype = float)
                        
                # We compute repulsion from walls
                
                wall_repulsion = np.array(agent.wall_repulsion(self.X_opt,self.Y_opt,self.Vs[agent.target]),dtype=float)
                
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
                    
        
        # After all agents have been summoned and their status updated, 
        # we advance time and simu step, to keep track of the time evolution of the simulation
        
        self.time+=dt
        self.simu_step+=1
        
        if verbose:
            print('t = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)+10*' ',end='\n')
            
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
    
    def run(self, verbose = False, draw = False, mode = 'scatter'):
        
     # Before starting the abm simulation, we compute the optimal velocities 
     # by solving the HJB representing obstacles and targets.
        
     for target in self.targets:
         self.targets[target].compute_optimal_velocity()
            
     while (self.inside > 0) & (self.time < self.T): 
         # The abm simulation is updated one step at the time, 
         # following the rules of the 'step' method
                
         self.step(self.dt,verbose = verbose)
                
         # Draw current state of the simulation 
                
         # and (int(self.time*100%10) == 0)
            
         if draw and (self.simu_step % 10) == 0 :
             self.draw(mode)
             plt.show()
                    
         # Print evacuation status
            
     if self.inside == 0:
        print('Evacuation complete in {:.2f}s!'.format(self.time))
     else:
        print('Evacuation failed!'+10*' ')
         
    # The 'gaussian_density' method computes the gaussian convolution of the agents
    # positions. Each position is the center of a gaussian with standard deviation
    # given by the 'sigma_convolution' paramter.      
    
    def gaussian_density(self,sigma,Nx,Ny):
     
        X,Y = np.meshgrid(np.linspace(0,self.room_length,self.Nx), 
                          np.linspace(0,self.room_height,self.Ny))
        
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
            
        d[self.V < 0] = 0
        
        return d
    
    # This method draws the trajectory of the agents at the end of the simulation
    
    def draw_final_trajectories(self):
        
        plt.figure(figsize = (self.room_length,self.room_height))
        colors = sns.color_palette(n_colors=len(self.targets))
        
        for i in range(self.N):
            agent = self.agents[i]
            color_index = list(self.targets).index(agent.target)
            traj = np.array(agent.traj,dtype = float)
            plt.plot(traj[:,0],traj[:,1],color = colors[color_index])
        
        plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])
        plt.title('{}/{} pedestrians evacuated in {:.2f}s'.format(self.N-self.inside,self.N,self.time))
        handles = [Patch(color=colors[i],label = list(self.targets)[i]) for i in range(len(self.targets))]
        plt.legend(handles = handles, loc = 'upper right', frameon = False)
        plt.show()
        
        
    # This module allows for the creation of the potential V
    # with a given target 
    
    def create_potential(self,var_room,var_config,targets):
        
        pot = self.pot
        V = np.zeros((self.Ny,self.Nx)) + pot
        V[1:-1,1:-1] = 0
        
        for walls in var_room['walls']:
            wall = var_room['walls'][walls]

            mask_X = abs(self.X_opt-wall[0]) < wall[2]/2
            mask_Y = abs(self.Y_opt-wall[1]) < wall[3]/2          
            
            V_temp = np.zeros((self.Ny,self.Nx))
            V_temp[mask_X*mask_Y] = pot
            
            V += V_temp
         
        for holes in var_room['holes']:
            hole = var_room['holes'][holes]
        
            hole_X = abs(self.X_opt-hole[0]) < hole[2]/2
            hole_Y = abs(self.Y_opt-hole[1]) < hole[3]/2
            
            V[hole_X*hole_Y] = 0   
        
        for cyls in var_room['cylinders']:
            cyl = var_room['cylinders'][cyls]
            
            V_temp =  np.zeros((self.Ny,self.Nx))
            V_temp[np.sqrt((self.X_opt-cyl[0])**2 + (self.Y_opt-cyl[1])**2) < cyl[2]] = pot
            
            V+= V_temp
        
        V = pot *(V <= pot)  
        V[:,0] = pot
        V[:,-1] = pot
        V[0,:] = pot
        V[-1,:] = pot
        
        for target in targets:
            target = var_room['targets'][target]
            target_X = abs(self.X_opt - target[0]) < target[2]/2
            target_Y = abs(self.Y_opt - target[1]) < target[3]/2
            V[target_X*target_Y] = var_config['hjb_params']['target_potential']
        
        return V