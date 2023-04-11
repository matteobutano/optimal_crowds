# author Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay

# Modules are imported 

import numpy as np
import matplotlib.pyplot as plt

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
        door_width_x,door_width_y = self.doors[self.target,2:4]
        return (x_door,y_door, door_width_x,door_width_y)
        
    def check_status(self):
        x,y = self.position()
        door = self.look_target()
        if abs(x-door[0]) < door[2]*0.5 and abs(y-door[1]) < door[3]*0.5  :
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
            rep = np.array((repulsion_intensity / distance )* rep_v,dtype = float)
        return rep
        
    def wall_repulsion(self,repulsion_radius, repulsion_intensity,X,Y,V):
        
        x,y = self.position()
    
        d = np.sqrt((X-x)**2 + (Y-y)**2)
        
        ind = np.unravel_index(np.argmin(d + V*10e10), d.shape)
        
        if d[ind] < repulsion_radius:
            rep = -np.array((x - X[ind],y-Y[ind]),dtype = float)
            return repulsion_intensity* (rep/d[ind])
        
        else:
            return np.array((0,0),dtype = float)
                    
    def draw_trajectory(self):
        traj = np.array(self.traj)
        plt.plot(traj[:,0],traj[:,1])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])