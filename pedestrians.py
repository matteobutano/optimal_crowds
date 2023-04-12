# author Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

# Modules are imported 

import numpy as np
import matplotlib.pyplot as plt

# The 'ped' class represent a pedestrian agent of the agent-based simulation. 
# This class allows for the monitoring of all the main features of a walking pedestrians
# and define the rules to compute the repulsion from others and from the 
# environment agents are subjected to.  
  
class ped:   
    def __init__(self,x,y,vx,vy,doors,room_length,room_height):
        
        # Parameters are passed as argument to allow greater agents personalization
        
        self.initial_position = np.array((x,y),dtype = float)
        self.initial_velocity = np.array((vx,vy),dtype = float)
        
        # The status determines wether an agent is still inside the room or not
        
        self.status = True
        
        # The target represent the favourite simulation exits
        
        self.target = 0
        
        # Agents keep track of time
        
        self.time = 0
        
        # Agents know the available escapes
        
        self.doors = doors
        
        # Agents know the room main features
        
        self.room_length = room_length
        self.room_height = room_height
        
        # Initialization of history of positions and velocities, 
        # for an eventual later analysis
        
        self.traj = []
        self.vels = []
        self.traj.append(self.initial_position)
        self.vels.append(self.initial_velocity)
    
    # This method identifies the actual target as the closest of doors
    
    def choose_target(self):
        x,y = self.position()
        distances = np.sqrt((x-self.doors[:,0])**2 + (y-self.doors[:,1])**2)
        self.target = np.argmin(distances)
        
    # This method acquire's the current target's main features
    
    def look_target(self):
        x_door,y_door = self.doors[self.target,:2]
        door_width_x,door_width_y = self.doors[self.target,2:4]
        return (x_door,y_door, door_width_x,door_width_y)
        
    # This method check wether the agent is still to be considered inside the room
    
    def check_status(self):
        x,y = self.position()
        door = self.look_target()
        if abs(x-door[0]) < door[2]*0.5 and abs(y-door[1]) < door[3]*0.5  :
            self.status = False
              
    # The following two methods give the present position and velocity of the agent
    
    def position(self):
        return np.array(self.traj[-1],dtype = float)
    
    def velocity(self):
        return np.array(self.vels[-1],dtype = float)
    
    # This method add a given position and velocity to the agent's history
    
    def evolve(self,x,y,vx,vy,dt):
        self.traj.append(np.array((x,y),dtype = float))
        self.vels.append(np.array((vx,vy),dtype = float))
        self.time += dt
    
    # The 'evac_time' method gives the evacuation time of an agent if he has already left the room
    
    def evac_time(self):
        if self.status:
            raise ValueError('This pedestrian has not exited the room yet!')
        return self.time
    
    def distance(self,x,y):
        return np.sqrt((self.position()[0]-x)**2 + (self.position()[1]-y)**2)
    
    # The following two methods compute the repulsion's intensity and direction. 
    
    def compute_repulsion(self,pos,repulsion_radius, repulsion_intensity):
        
        # Given a position, (usually another agent's), the repulsion is computed as the 
        # vector originating from that position directed to the agent's current pos
        # centered in the agent's current position. 
        
        rep_v = -self.position() + pos
        
        distance = np.sqrt(rep_v[0]**2 + rep_v[1]**2)
        
        rep = np.array((0,0),dtype = float)
        
        # Repulsion from others kicks in only under a threshold
        
        if distance < repulsion_radius:
            rep = np.array((repulsion_intensity / distance )* rep_v,dtype = float)
            
        return rep
        
    def wall_repulsion(self,repulsion_radius, repulsion_intensity,X,Y,V):
        
        # In order to compute walls repulsion, we calculate the closest point
        # belonging to the potential V and compute the vector along that direction pointing
        # away from the door. 
        
        x,y = self.position()
    
        d = np.sqrt((X-x)**2 + (Y-y)**2)
        
        ind = np.unravel_index(np.argmin(d + V*10e10), d.shape)
        
        # Repulsion from the walls kicks in only under a threshold 
        
        if d[ind] < repulsion_radius:
            rep = -np.array((x - X[ind],y-Y[ind]),dtype = float)
            return repulsion_intensity* (rep/d[ind])
        
        else:
            return np.array((0,0),dtype = float)
                    
    # This method draws an agent's full trajectory 
    
    def draw_trajectory(self):
        traj = np.array(self.traj)
        plt.plot(traj[:,0],traj[:,1])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])