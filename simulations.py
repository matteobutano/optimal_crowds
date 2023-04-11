import numpy as np
import matplotlib.pyplot as plt
from optimal_crowds import pedestrians
from optimal_crowds import optimals
import json

# Create class to describe simulation

class simulation:
    def __init__(self,room, mode,T = 10):
        with open('optimal_crowds/config.json') as f:
            var_config = json.loads(f.read())
            
        with open('rooms/'+room+'.json') as f:
            var_room = json.loads(f.read())
        
        # Read doors
        self.doors = np.empty((len(var_room['doors']),5))
        for i,door in enumerate(var_room['doors']):
            self.doors[i,:] = np.array(var_room['doors'][str(door)])
        
        # Read room 
        self.room_length = var_room['room_length']
        self.room_height = var_room['room_height']
        self.pot = var_config['hjb_params']['potential']
        
        self.Nx = var_room['Nx']
        self.Ny = var_room['Ny']
        self.sigma_convolution = var_config['sigma_convolution']
        
             
        self.dx = self.room_length/(self.Nx-1)
        self.dy = self.room_height/(self.Ny-1)

        self.X_opt, self.Y_opt = np.meshgrid(np.linspace(0,self.room_length,self.Nx)
                                             ,np.linspace(0,self.room_height,self.Ny))
        
        self.u_0 = np.zeros((self.Ny,self.Nx),dtype = float)
        self.phi_0 =  np.zeros((self.Ny,self.Nx),dtype = float)
        self.evacuator = np.zeros((self.Ny,self.Nx),dtype = float) + 1
        
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
        
        for door in var_room['doors']:
            door = var_room['doors'][door]
            
            door_X = abs(self.X_opt - door[0]) < door[2]/2
            door_Y = abs(self.Y_opt - door[1]) < door[3]/2
           
            self.evacuator[door_X*door_Y] = 0
            self.V[door_X*door_Y] = door[4]
            
        self.phi_0 = self.phi_0.reshape(self.Nx*self.Ny)
        
        # Init time variables 
        self.T = T
        self.time = 0.
        self.simu_step = 0
        self.dt = var_room['dt']
        
        # Init population
        self.initial_density = var_room['initial_density']
        self.initial_box = var_room['initial_box']
        self.N = int(self.initial_density*(self.initial_box[1]-self.initial_box[0])* (self.initial_box[3]-self.initial_box[2]))
        self.inside = self.N
         
        # Init pedestrians features
        self.relaxation = var_config['relaxation']
        self.rep_radius = var_config['repulsion_radius']
        self.rep_int = var_config['repulsion_intensity']
        self.noise_intensity = var_config['hjb_params']['sigma']
        self.des_v = var_config['des_v']
        
        self.agents = np.empty(self.N,dtype=object)
     
        self.m_0 = np.zeros((self.Ny,self.Nx),dtype = float) 
            
        self.type = mode
        
        self.optimal = optimals.optimal_trajectories(room,T)
        
        if self.type == 'abm':
           # Create crowd
           xs = np.random.uniform(self.initial_box[0], self.initial_box[1],self.N)
           ys = np.random.uniform(self.initial_box[2],self.initial_box[3],self.N)
           for i in range(self.N):
               self.agents[i] = pedestrians.ped(xs[i], ys[i], 0, 0, self.doors, self.room_length, self.room_height)
               self.agents[i].choose_target()
               
           print('ABM simulation room created!')
                     
           self.optimal.compute_optimal_velocity()
        
        elif self.type =='mfg':
           x_min = self.initial_box[0]
           x_max = self.initial_box[1]
           y_min = self.initial_box[2]
           y_max = self.initial_box[3]
           X = self.X_opt
           Y = self.Y_opt
           self.m_0[((X > x_min) & (X < x_max)) * ((Y > y_min) & (Y < y_max))] = self.initial_density 
        
           print('MFG simulation room created!')
        
    def draw(self,mode = 'scatter'):
        
        plt.figure(figsize = (self.room_length,self.room_height))
        
        if self.type == 'abm':
        
            if mode == 'scatter':
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
                d = self.gaussian_density(self.sigma_convolution, self.Nx, self.Ny)
                # plt.imshow(d.reshape(self.Ny,self.Nx),extent=[0,self.room_length,0,self.room_height])
                
                plt.imshow(np.flip(self.V/self.pot,axis = 0) + d,extent=[0,self.room_length,0,self.room_height])
                plt.xlim([0,self.room_length])
                plt.ylim([0,self.room_height])
                plt.colorbar()
                plt.clim(0,2)
                title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
                plt.title(title)
                plt.show()
            
        elif self.type == 'mfg':
            
            plt.pcolor(self.X_opt, self.Y_opt,self.m_0 + 2*self.initial_density*self.V/self.pot)
                    
        
            
    def step(self,dt,verbose =  False):
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            # Summon agent
            agent = self.agents[i]
    
            # Check if agent has reached the a door
            if agent.status:
            
                # Compute desired velocity 
                des_x, des_y  = self.optimal.choose_optimal_velocity(agent.position(), self.simu_step)
                
                # Compute repulsion from nearby pedestrians
                repulsion = np.array((0, 0),dtype = float)
                for j in range(self.N):
                    if self.agents[j].status and j != i:
                        repulsion = repulsion + np.array(agent.compute_repulsion(self.agents[j].position(), self.rep_radius, self.rep_int),dtype = float)
                        
                # Compute repulsion from walls
                wall_repulsion = np.array(agent.wall_repulsion(self.rep_radius, self.rep_int,self.X_opt,self.Y_opt,self.V),dtype=float)
                
                # Compute current velocity with random perturbation and repulsion
                current_velocity = agent.velocity() + 0.5*self.noise_intensity*np.random.normal(size = 2) + repulsion + wall_repulsion 
        
                    
                # Compute acceleration towards desired velocity
                ax = (des_x - current_velocity[0]) / self.relaxation
                ay = (des_y - current_velocity[1]) / self.relaxation
            
                # Compute new position and velocity
                old_x = agent.position()[0]
                old_y = agent.position()[1]
                x = old_x + current_velocity[0] * dt + 0.5 * ax * dt**2
                y = old_y + current_velocity[1] * dt + 0.5 * ay * dt**2
                vx = current_velocity[0] + ax * dt
                vy = current_velocity[1] + ay * dt
                
                # Update agent position
                agent.evolve(x, y, vx, vy, dt)
                
                # Check if agent has left the room
                agent.check_status()
                if agent.status == False:
                    self.inside +=-1
                    
                # Choose new target
                
                agent.choose_target()
        
        # Avance time and simu step
        self.time+=dt
        self.simu_step+=1
        
        if verbose:
            print('t = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N),end='\n')
                
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
            
    def initial_positions(self):
        xs = np.empty(self.N,dtype = float)
        ys = np.empty(self.N,dtype = float)
        for i, agent in enumerate(self.agents):
            xs[i] = agent.initial_position[0]
            ys[i] = agent.initial_position[1]
        return xs,ys

    def run(self, verbose = True, draw = False, mode = 'scatter'):
        
        if self.type == 'abm':
        
            while (self.inside > 0) & (self.time < self.T):    
                self.step(self.dt,verbose = verbose)
                if draw:
                    self.draw(mode)
                    plt.show()
            if self.inside == 0:
                print('Evacuation complete!')
            else:
                print('Evacuation failed!')
            
        elif self.type == 'mfg':
            
            self.optimal.mean_field_game(self.m_0,draw = draw,verbose=verbose)
                
    def gaussian_density(self,sigma,Nx,Ny):
        
        dx = self.room_length/(Nx)
        dy = self.room_height/(Ny)

        X,Y = np.meshgrid(np.linspace(0,self.room_length,Nx+1), 
                          np.linspace(0,self.room_height,Ny+1))
        
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