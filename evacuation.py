import numpy as np
import matplotlib.pyplot as plt
import json

# Create class to describe simulation

class simulation:
    def __init__(self,N,dt):
        with open('abm_evacuation/config.json') as f:
            var = json.loads(f.read())
        
        # Read doors
        self.doors = np.empty((len(var['doors']),4))
        for i,door in enumerate(var['doors']):
            self.doors[i,:] = np.array(var['doors'][str(door)])
        
        # Read room 
        self.room_length = var['room_length']
        self.room_height = var['room_height']
        
        # Read params for density plot
        self.Nx = var['Nx']
        self.Ny = var['Ny']
        self.sigma = var['sigma']
        
        # Init time variables 
        self.time = 0.
        self.simu_step = 0
        self.dt = dt
        
        # Init population
        self.N = N
        self.inside = N
         
        # Init pedestrians features
        self.relaxation = var['relaxation']
        self.rep_radius = var['repulsion_radius']
        self.rep_int = var['repulsion_intensity']
        self.noise_intensity = var['noise_intensity']
        self.des_v = var['des_v']
        
        # Create crowd
        self.agents = np.empty(N,dtype=object)
        xs = np.random.uniform(self.rep_radius, self.room_length-self.rep_radius,N)
        ys = np.random.uniform(self.rep_radius, self.room_height-self.rep_radius,N)
        for i in range(N):
            self.agents[i] = ped(xs[i], ys[i], 0, 0, self.doors, self.room_length, self.room_height)
            self.agents[i].choose_target()
        print('Simulation room created!')
        
    def draw(self,mode):
        if mode == 'scatter':
            scat_x = [self.agents[i].position()[0] for i in range(self.N) if self.agents[i].status]
            scat_y = [self.agents[i].position()[1] for i in range(self.N) if self.agents[i].status]
            plt.scatter(scat_x,scat_y,color = 'blue')
            for door in self.doors:
                plt.plot([door[0]-door[2]/2, door[0] + door[2]/2],[door[1] - door[3]/2, door[1] + door[3]/2], 'r-', linewidth=4)
            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            print_time = '{:.2f}'.format(self.time)
            print_fraction = '{:.2f}'.format(100. - 100.*self.inside/self.N)
            title = 't = '+ print_time +' s, evac = '+print_fraction+' %'
            plt.title(title)
        if mode == 'density':
            X,Y,d = self.gaussian_density(self.sigma, self.Nx, self.Ny)
            plt.pcolor(X,Y,d) 
            for door in self.doors:
                plt.plot([door[0]-door[2]/2, door[0] + door[2]/2],[door[1] - door[3]/2, door[1] + door[3]/2], 'r-', linewidth=4)
            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            print_time = '{:.2f}'.format(self.time)
            print_fraction = '{:.2f}'.format(100. - 100.*self.inside/self.N)
            title = 't = '+ print_time +' s, evac = '+print_fraction+' %'
            plt.title(title)
            plt.colorbar()
            
    
    def step(self,dt,verbose =  False):
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            # Summon agent
            agent = self.agents[i]
    
            # Check if agent has reached the a door
            if agent.status:
            
                # Compute desired velocity 
                des_x,des_y = agent.desired_velocity(self.des_v)
                
                # Compute repulsion from nearby pedestrians
                repulsion = np.array((0, 0),dtype = float)
                for j in range(self.N):
                    if self.agents[j].status and j != i:
                        repulsion = repulsion + np.array(agent.compute_repulsion(self.agents[j].position(), self.rep_radius, self.rep_int),dtype = float)
                        
                # Compute repulsion from walls
                wall_repulsion = np.array(agent.wall_repulsion(self.rep_radius, self.rep_int),dtype=float)
                
                # Compute current velocity with random perturbation and repulsion
                current_velocity = agent.velocity() + np.random.uniform(-self.noise_intensity, self.noise_intensity,2) + repulsion + wall_repulsion 
        
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

        
    
    def run(self,save = False, where_save = '', verbose = False, draw = False,mode = 'scatter'):
        if save:
            densities = []
        while self.inside > 0:    
            self.step(self.dt,verbose = verbose)
            if draw:
                self.draw(mode)
                plt.show()
            if save:
                densities.append(self.gaussian_density(self.sigma, self.Nx, self.Ny)[2])
        print('Evacuation complete!')
        if save:
            densities = np.array(densities).reshape((self.simu_step,self.Nx*self.Ny))
            np.savetxt(where_save+'m_dt='+str(self.dt)+'_N='+str(self.N)+'.txt', densities)
            print('Densities saved!')
                
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
        return X,Y,d
        
# Create class to describe pedestrian 
  
class ped:   
    def __init__(self,x,y,vx,vy,doors,room_length,room_height):
        self.initial_position = np.array((x,y),dtype = float)
        self.initial_velocity = np.array((vx,vy),dtype = float)
        self.status = True
        self.target = 0
        self.time = 0
        self.doors = doors
        self.room_length = room_length
        self.room_height = room_height
        self.traj = []
        self.vels = []
        self.traj.append(self.initial_position)
        self.vels.append(self.initial_velocity)
    
    def choose_target(self):
        x,y = self.position()
        distances = np.sqrt((x-self.doors[:,0])**2 + (y-self.doors[:,1])**2)
        self.target = np.argmin(distances)
        
    def look_target(self):
        x_door,y_door = self.doors[self.target,:2]
        door_width_x,door_width_y = self.doors[self.target,2:]
        return (x_door,y_door, door_width_x,door_width_y)
        
        
    def check_status(self):
        x,y = self.position()
        door = self.look_target()
        if abs(x-door[0]) < door[2]*0.5 + door[3]*0.2 and abs(y-door[1]) < door[3]*0.5 + door[2]*0.2 :
            self.status = False
              
    def position(self):
        return np.array(self.traj[-1],dtype = float)
    
    def velocity(self):
        return np.array(self.vels[-1],dtype = float)
    
    def evolve(self,x,y,vx,vy,dt):
        self.traj.append(np.array((x,y),dtype = float))
        self.vels.append(np.array((vx,vy),dtype = float))
        self.time += dt
    
    def evac_time(self):
        if self.status:
            raise ValueError('This pedestrian has not exited the room yet!')
        return self.time
    
    def distance(self,x,y):
        return np.sqrt((self.position()[0]-x)**2 + (self.position()[1]-y)**2)
    
    def desired_velocity(self,des_v):
        x_door,y_door = self.doors[self.target,:2]
        return des_v*((-self.position() + np.array((x_door,y_door),dtype = float))/self.distance(x_door,y_door))
    
    def compute_repulsion(self,pos,repulsion_radius, repulsion_intensity):
        rep_v = -self.position() + pos
        distance = np.sqrt(rep_v[0]**2 + rep_v[1]**2)
        rep = np.array((0,0),dtype = float)
        if distance < repulsion_radius:
            rep = np.array(((repulsion_intensity * (repulsion_radius - distance)) / distance )* rep_v,dtype = float)
        return rep
        
    def wall_repulsion(self,repulsion_radius, repulsion_intensity):
        
        wall_repulsion = np.array((0, 0),dtype = float)
        
        x,y = self.position()
        
        door = self.look_target()

        if x < repulsion_radius and abs(y-door[1]) > door[3]/2:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - x) / repulsion_radius * np.array((1, 0))
            
        elif x > self.room_length - repulsion_radius and abs(y-door[1]) > door[3]/2:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (self.room_length - x)) / repulsion_radius * np.array((-1, 0))
            
        if y < repulsion_radius and abs(x-door[0]) > door[2]/2:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - y) / repulsion_radius * np.array((0, 1))
            
        elif y > self.room_height - repulsion_radius and abs(x-door[0]) > door[2]/2:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (self.room_height - y)) / repulsion_radius * np.array((0, -1))
            
        return wall_repulsion
            
    def draw_trajectory(self):
        traj = np.array(self.traj)
        plt.plot(traj[:,0],traj[:,1])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])
        
