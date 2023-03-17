import numpy as np
import matplotlib.pyplot as plt
import json

# Create class to describe simulation

class simulation:
    def __init__(self,N,dt):
        with open('config.json') as f:
            var = json.loads(f.read())
        self.x_door = var['x_door']
        self.y_door = var['y_door']
        self.door_width = var['door_width'] 
        self.room_length = var['room_length']
        self.room_height = var['room_height']
        self.time = 0.
        self.simu_step = 0
        self.dt = dt
        self.N = N
        self.inside = N
        self.relaxation = var['relaxation']
        self.rep_radius = var['repulsion_radius']
        self.rep_int = var['repulsion_intensity']
        self.noise_intensity = var['noise_intensity']
        self.des_v = var['des_v']
        self.agents = np.empty(N,dtype=object)
        xs = np.random.uniform(self.rep_radius, self.room_length-self.rep_radius,N)
        ys = np.random.uniform(self.rep_radius, self.room_height-self.rep_radius,N)
        for i in range(N):
            self.agents[i] = ped(xs[i], ys[i], 0, 0, self.x_door, self.y_door, self.door_width, self.room_length, self.room_height)
        print('Simulation room created!')
        
    def draw(self,mode):
        if mode == 'scatter':
            scat_x = [self.agents[i].position()[0] for i in range(self.N) if self.agents[i].status]
            scat_y = [self.agents[i].position()[1] for i in range(self.N) if self.agents[i].status]
            plt.scatter(scat_x,scat_y,color = 'blue')
            plt.plot([self.x_door-self.door_width/2, self.x_door + self.door_width/2],[0, 0], 'r-', linewidth=2)
            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            print_time = '{:.2f}'.format(self.time)
            print_fraction = '{:.2f}'.format(100. - 100.*self.inside/self.N)
            title = 't = '+ print_time +' s, evac = '+print_fraction+' %'
            plt.title(title)
    
    def step(self,dt):
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            # Summon agent
            agent = self.agents[i]
            
            # Check if agent has reached the door
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
        
        # Avance time and simu step
        self.time+=dt
        self.simu_step+=1
        print('t = {:.2f}s exit = {:.2f}'.format(self.time,float(100 - self.inside/self.N*100)))
                
    def evac_times(self,draw = False):
        if self.inside > 0:
            raise ValueError('There are still people inside!')
        times = np.empty(self.N,dtype = float)
        xs = np.empty(self.N)
        ys = np.empty(self.N)
        for i, agent in enumerate(self.agents):
            if draw:
                xs[i] = agent.initial_position[0]
                ys[i] = agent.initial_position[1]
            times[i] = agent.time
        if draw: 
            plt.scatter(xs, ys, c=times)
            plt.xlim(0,self.room_length)
            plt.ylim(0,self.room_height)
            title = 'Evacuation time'
            plt.title(title)
            plt.colorbar()
        else:
            return times
            
            
        
    
    def run(self,draw = False,mode = 'scatter'):
        while self.inside > 0:    
            self.step(self.dt)
            if draw:
                self.draw(mode)
                plt.show()
        print('Evacuation complete!')        
                
    def gaussian_density(self,sigma,Nx,Ny):
        
        def gaussian(x,y,x_agent,y_agent,sigma):
            c_x = x - x_agent
            c_y = y - y_agent
            C = np.sqrt(4*np.pi**2*sigma**2)
            return np.exp(-(c_x**2 + c_y**2)/(2*sigma**2))/C
        
        dx = self.room_length/(Nx-1)
        dy = self.room_height/(Ny-1)

        X,Y = np.meshgrid(np.linspace(0,self.room_length,Nx+1), 
                          np.linspace(0,self.room_height,Ny+1))
        
        X = X[:-1,:-1] + dx/2
        Y = np.flip(Y[:-1,:-1] + dy/2,axis = 0)
        
        density = np.zeros((Ny,Nx))
        
        for agent in self.agents:
            density += gaussian(X, Y, agent[0], agent[1], 0.1)
        
        return density
        
# Create class to describe pedestrian 
  
class ped:   
    def __init__(self,x,y,vx,vy,x_door,y_door,door_width,room_length,room_height):
        self.initial_position = np.array((x,y),dtype = float)
        self.initial_velocity = np.array((vx,vy),dtype = float)
        self.status = True
        self.time = 0
        self.x_door = x_door
        self.y_door = y_door
        self.door_width = door_width 
        self.room_length = room_length
        self.room_height = room_height
        self.traj = []
        self.vels = []
        self.traj.append(self.initial_position)
        self.vels.append(self.initial_velocity)
   
    def check_status(self):
        x = self.position()[0]
        y = self.position()[1]
        if ((y < 0.1) and (self.x_door - self.door_width/2 < x < self.x_door + self.door_width/2)):
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
    
    def distance(self):
        return np.sqrt((self.position()[0]-self.x_door)**2 + (self.position()[1]-self.y_door**2))
    
    def desired_velocity(self,des_v):
        return des_v*((-self.position() + np.array((self.x_door,self.y_door),dtype = float))/self.distance())
    
    def compute_repulsion(self,pos,repulsion_radius, repulsion_intensity):
        rep_v = -self.position() + pos
        distance = np.sqrt(rep_v[0]**2 + rep_v[1]**2)
        rep = np.array((0,0),dtype = float)
        if distance < repulsion_radius:
            rep = np.array(((repulsion_intensity * (repulsion_radius - distance)) / distance )* rep_v,dtype = float)
        return rep
        
    def wall_repulsion(self,repulsion_radius, repulsion_intensity):
        wall_repulsion = np.array((0, 0),dtype = float)
        x = self.position()[0]
        y = self.position()[1]
        if x < repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - x) / repulsion_radius * np.array((1, 0))
        elif x > self.room_length - repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (self.room_length - x)) / repulsion_radius * np.array((-1, 0))
        if y < repulsion_radius and (x > self.x_door + self.door_width/2 or x < self.x_door - self.door_width/2) :
            wall_repulsion += -repulsion_intensity*(repulsion_radius - y) / repulsion_radius * np.array((0, 1))
        elif y > self.room_height - repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (self.room_height - y)) / repulsion_radius * np.array((0, -1))
        return wall_repulsion
            
    
        
