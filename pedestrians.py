# author: Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

# Modules are imported 

import numpy as np

# The 'ped' class represent a pedestrian agent of the agent-based simulation. 
# This class allows for the monitoring of all the main features of a walking pedestrians
# and define the rules to compute the repulsion from others and from the 
# environment agents are subjected to.  
  
class ped:   
    def __init__(self,X,Y,step,V,target,all_targets,possible_targets,x,y,vx,vy,room_length,room_height,v_des,a_min,tau_a,b_min,b_max,eta):
        '''
        Creates one pedestrian object.

        Parameters
        ----------
        X : numpy.array
            x coordinates of the simulation grid.
        Y : numpy.array
            y coordinates of the simulation grid.
        step : float
            Size of the grid step.
        V : numpy.array
            Matrix containing the potential.
        target : str
            Name of target.
        all_targets : dict 
            Dictionary of all targets in the room.
        possible_targets : dict
            List of available target for the this pedestrian.
        x : float
            Inital position's x coordinate.
        y : float
            Inital position's y coordinate.
        vx : float
            Inital velocity's x coordinate.
        vy : float
            Inital velocity's y coordinate.
        room_length : float
            Length of the room.
        room_height : float
            Height of the room.
        v_des : float
            Desired speed.
        a_min : float
            Min value of the agent's ellypse major semiax.
        tau_a : float
            Incremental factor of ellypse major semiax.
        b_min : float
            Min value of the agent's ellypse minor semiax.
        b_max : float
            Max value of the agent's ellypse minor semiax.
        eta : float
            Repulsion intensity.

        Returns
        -------
        None.

        '''
        
        # Parameters are passed as argument to allow greater agents personalization
        
        self.initial_position = np.array((x,y),dtype = float)
        self.initial_velocity = np.array((vx,vy),dtype = float)
        
        # Repulsion parameters 
        
        self.v_des = v_des
        self.a_min = b_min
        self.tau_a = tau_a
        self.b_min = b_min
        self.b_max = b_max
        self.eta = eta
        
        # The status determines wether an agent is still inside the room or not
        
        self.status = True
        
        # The target represent the favourite simulation exits
        
        self.target = target
        
        # The agent knows all the exits
        
        self.possible_targets = possible_targets
        self.all_targets = all_targets
        
        # Agents keep track of time
        
        self.time = 0
        
        # Agents know the room main features
        
        self.room_length = room_length
        self.room_height = room_height
        
        # Initialization of history of positions and velocities, 
        # for an eventual later analysis
        
        self.traj = []
        self.vels = []
        
        self.traj.append(self.initial_position)
        self.vels.append(self.initial_velocity)
        
        # The room is given aswell
        
        self.X = X
        self.Y = Y
        self.V = V
        self.step = step
        
    # This method check wether the agent is still to be considered inside the room
    
    def check_status(self):
        '''
        Check if ped is still in the room. If it is not, it changes the flag.

        Returns
        -------
        None.

        '''
        x,y = self.position()
        
        for target in self.possible_targets:
            door = self.all_targets[target]
            
            if abs(x-door[0]) < door[2]*0.5 and abs(y-door[1]) < door[3]*0.5  :
                self.status = False
              
    # The following two methods give the present position and velocity of the agent
    
    def position(self):
        '''
        Return ped current position.

        Returns
        -------
        numpy.array
            Ped current position.

        '''
        return np.array(self.traj[-1],dtype = float)
    
    def velocity(self):
        '''
        Return ped current velocity.

        Returns
        -------
        numpy.array
            Ped current velocity.

        '''
        return np.array(self.vels[-1],dtype = float)
    
    # This method add a given position and velocity to the agent's history
    
    def evolve(self, x, y, vx, vy, dt):
        '''
        Update pedestrian main parameters.

        Parameters
        ----------
        x : float
            New agent's position x coordinate.
        y : float
            New agent's position y coordinate.
        vx : float
            New agent's velocity x coordinate.
        vy : float
            New agent's velocity y coordinate.
        dt : float
            Time step.

        Returns
        -------
        None.

        '''
        
        self.traj.append(np.array((x,y),dtype = float))
        self.vels.append(np.array((vx,vy),dtype = float))
        self.time += dt
    
    # The 'evac_time' method gives the evacuation time of an agent if he has already left the room
    
    def evac_time(self):
        '''
        Return the time needed to evacuate the room.

        Raises
        ------
        ValueError
            Pedestrian must have reached the exit to have an exit time.

        Returns
        -------
        float
            Total time needed to exit room.

        '''
        if self.status:
            raise ValueError('This pedestrian has not exited the room yet!')
        return self.time
    
    # The following two methods compute the repulsion's intensity and direction. 
    
    def agents_repulsion(self,pos_j,vel_j):
        '''
        Compute the repulsion caused the presence of pedestrian j.

        Parameters
        ----------
        pos_j : numpy.array
            Coordinates of pedestrian j's position.
        vel_j : numpy.array
            Pedestrian j's velocity.

        Returns
        -------
        numpy.array
            Repulsion vector.

        '''
        
        # Given a position, (usually another agent's), the repulsion is computed
        # generalized centrifugal force model. 
        
        dot = lambda a,b: a[0]*b[0] + a[1]*b[1]
    
        pos_i = self.position()
        vel_i = self.velocity()
        
        a_i = self.a_min + self.tau_a * np.linalg.norm(vel_i)
        b_i = self.b_max - (self.b_max - self.b_min)*np.minimum(np.linalg.norm(vel_i)/self.v_des,1)
        a_j = self.a_min + self.tau_a * np.linalg.norm(vel_j)
        b_j = self.b_max - (self.b_max - self.b_min)*np.minimum(np.linalg.norm(vel_j)/self.v_des,1)
        
        # First we compute the relative position of j
        
        R = pos_j - pos_i
        e = R/np.linalg.norm(R)
        
        # Then we compute the relative velocity of j
      
        v_rel = vel_j - vel_i
        v_rel = 0.5*(dot(v_rel,-e) + abs(dot(v_rel,-e)))
      
        # We restrict the repulsion to what happens around the agent at 180°
        
        k = 0
        
        if np.linalg.norm(vel_i) > 0:
            k = np.maximum(dot(vel_i,e)/np.linalg.norm(vel_i) - np.cos(0.7*np.pi),0)/(1- np.cos(0.7*np.pi))
         
        # Now we compute the distance between ellypses along direction of the centers
    
        alpha_i = np.arctan2(R[1],R[0])
        beta_i = np.arctan2(vel_i[1],vel_i[0])
        alpha_j = np.arctan2(-R[1],-R[0])
        beta_j = np.arctan2(vel_j[1],vel_j[0])
        q_i = np.sqrt(1/((np.cos(alpha_i-beta_i)/a_i)**2 + (np.sin(alpha_i-beta_i)/b_i)**2))
        q_j = np.sqrt(1/((np.cos(alpha_j-beta_j)/a_j)**2 + (np.sin(alpha_j-beta_j)/b_j)**2))
        dist = np.linalg.norm(R) - q_i - q_j 
        
    
        # Finally we compute the repulsion, that kicks in only when two ellypses 
        # super pose
       
        rep = np.minimum(k * np.exp(- dist/(self.eta*(1 + v_rel))),1)
       
        return -rep*R
        
    def wall_repulsion(self,X,Y,V):
        '''
        Compute the repulsion cause by walls and obstacles

        Parameters
        ----------
        X : numpy.array
            Grid x coordinates.
        Y : numpy.array
            Grid y coordinates.
        V : numpy.array
            Potential.

        Returns
        -------
        numpy.array
            Repulsion vector.

        '''
        
        # In order to compute walls repulsion, we calculate the closest point
        # belonging to the potential V and compute the vector along that direction pointing
        # away from the door. 
        
        dot = lambda a,b: a[0]*b[0] + a[1]*b[1]
        
        pos_i = self.position()
        vel_i = self.velocity()
        
        distances = np.sqrt((X-pos_i[0])**2 + (Y-pos_i[1])**2)
        
        ind = np.unravel_index(np.argmin(distances + V*10e10), distances.shape)
        
        pos_wall = (X[ind],Y[ind])
        
        R = pos_wall - pos_i 
        e = R/np.linalg.norm(R)
        
        v_rel = 0.5*(dot(vel_i,e) + abs(dot(vel_i,e)))
       
        alpha_i = np.arctan2(R[1],R[0])
        beta_i = np.arctan2(vel_i[1],vel_i[0])
        
        a_i = self.a_min + self.tau_a * np.linalg.norm(vel_i)
        b_i = self.b_max - (self.b_max - self.b_min)*np.minimum(np.linalg.norm(vel_i)/self.v_des,1)
        
        q_i = 1/((np.cos(alpha_i-beta_i)/a_i)**2 + (np.sin(alpha_i-beta_i)/b_i)**2)
        
        dist = np.linalg.norm(R) - 2*q_i 
        
        rep = np.minimum(np.exp(-dist/(self.eta*(1 + v_rel))),1)
        
        return np.array(-rep*R,dtype=float)
    
    def distance(self,pos):
        '''
        Compute the distance between the pedestrian and a given point in space.

        Parameters
        ----------
        pos : numpy.array
            Given point in space.

        Returns
        -------
        float
            Distance.

        '''
        
        x,y = self.position()
        
        d = np.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)
        
        return d