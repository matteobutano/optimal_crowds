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
    def __init__(self,x,y,vx,vy,doors,room_length,room_height,des_v,a_min,tau_a,b_min,b_max,eta):
        
        # Parameters are passed as argument to allow greater agents personalization
        
        self.initial_position = np.array((x,y),dtype = float)
        self.initial_velocity = np.array((vx,vy),dtype = float)
        
        # Repulsion parameters 
        
        self.des_v = des_v
        self.a_min = b_min
        self.tau_a = tau_a
        self.b_min = b_min
        self.b_max = b_max
        self.eta = eta
        
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
    
    def compute_repulsion(self,pos_j,vel_j):
        
        def dis(pos_i, pos_j,vel,a,b):
            
            R = pos_j - pos_i
            alpha = np.arctan2(R[1],R[0])
            beta = np.arctan2(vel[1],vel[0])
            q = (np.cos(alpha-beta)/a)**2 + (np.sin(alpha-beta)/b)**2

            return 1/np.sqrt(q)
        
        # Given a position, (usually another agent's), the repulsion is computed
        # generalized centrifugal force model. 
        
        dot = lambda a,b: a[0]*b[0] + a[1]*b[1]

        pos_i = self.position()
        vel_i = self.velocity()
        
        a_i = self.a_min + self.tau_a * np.linalg.norm(vel_i)
        b_i = self.b_max - (self.b_max - self.b_min)*np.minimum(np.linalg.norm(vel_i)/self.des_v,1)
        a_j = self.a_min + self.tau_a * np.linalg.norm(vel_j)
        b_j = self.b_max - (self.b_max - self.b_min)*np.minimum(np.linalg.norm(vel_j)/self.des_v,1)
        
        
        # First we compute the relative position of j
        
        R = pos_j - pos_i
        e = R/np.linalg.norm(R)
        
        # Then we compute the relative velocity of j
        
        v_rel = 0 
        
        if np.linalg.norm(vel_i - vel_j) > 0:
    
            v_rel = vel_i - vel_j
            
            v_rel = 0.5*(dot(v_rel,e) + abs(dot(v_rel,e)))
            
            v_rel = v_rel/np.linalg.norm(vel_i - vel_j)
        
        # We restrict the repulsion to what happens around the agent at 180°
        
        k = 0
        
        if np.linalg.norm(vel_i) > 0:
            k = 0.5*(dot(vel_i,e) + abs(dot(vel_i,e)))/np.linalg.norm(vel_i)
        
        # Now we compute the distance between ellypses along direction of the centers
        
        d_i = dis(pos_i,pos_j,vel_i, a_i, b_i)
        d_j = dis(pos_j,pos_i,vel_j, a_j, b_j)
        dist = np.linalg.norm(R) -d_i-d_j 
        rep = -self.eta*k*(v_rel)**2/dist
        
        # plt.arrow(pos_i[0], pos_i[1],rep*R[0], rep*R[1])
        # plt.xlim([0,5])
        # plt.ylim([0,5])
        # plt.show()
        
        # print('k = {:.2f}, dist =  {:.2f}, v_rel =  {:.2f}, rep =  {:.2f}'.format(k,dist,v_rel,rep))
        
        return rep*R
        
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