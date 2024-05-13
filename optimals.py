# author: Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json


class optimals:
    def __init__(self, room, V, T, target):
        """
        Initialise the optimals object, containing the velocities prescribed as
        per the solution of HJB associated to the simulation.

        Parameters
        ----------
        room : str
            name without extension of the .json file containing the room's description.
        V : numpy.array
            Matrix containing the potential describing the room walls and obstacles.
        T : float
            time limit for the evacuation.
        target : list
            List of targets.

        Returns
        -------
        None.

        """
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
        
        # A grid is defined for the MFG equations
        
        self.grid_step = var_config['grid_step']
        
        self.Nx = int(self.room_length//self.grid_step + 1)
        self.Ny = int(self.room_height//self.grid_step + 1)
        
        self.dx = self.grid_step
        self.dy = self.grid_step

        self.X_opt, self.Y_opt = np.meshgrid(np.linspace(0,self.room_length,self.Nx)
                                             ,np.linspace(0,self.room_height,self.Ny))
        
        # Read optimization parameters
        
        self.sigma = var_config['hjb_params']['sigma']
        self.mu = var_config['hjb_params']['mu']
        self.pot = var_config['hjb_params']['wall_potential']
        self.pot_target = var_config['hjb_params']['target_potential']
        self.g = var_config['hjb_params']['g']
   
        # Time discretization 
        
        self.dt = var_config['dt']
        self.T = T
        self.nt_opt = round(self.T/self.dt)
            
        # Create boundary conditions and final cost
         
        self.vx_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        self.vy_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        
        self.phi_T =  np.zeros((self.Ny,self.Nx),dtype = float) + 1
        
        self.target = target
        
        # Create potential V
        
        self.V = V
        self.V[self.V<0] = self.pot
        self.V[self.V>0] = self.pot_target
            
        self.phi_T = self.phi_T.reshape(self.Nx*self.Ny)
        
        self.lim = 10e-3
           
    def draw_optimal_velocity(self):
        """
        Quiver plot of HJB velocity for each point of the grid.

        Returns
        -------
        None.

        """
        for i in range(self.nt_opt-1):

            if i < self.nt_opt-2:
                plt.quiver(self.X_opt[1:-1, 1:-1], self.Y_opt[1:-1, 1:-1],
                           self.vx_opt[i], self.vy_opt[i])
            else:
                plt.plot()

            plt.xlim([0,self.room_length])
            plt.ylim([0,self.room_height])
            title = 't = {:.2f}s'.format(i*self.dt)
            plt.title(title)
            plt.show()
    
    # The 'compute_optimal_velocity' method computes the HJB equation 
    # over the simulation room using the potential V to represent the walls
    # and the doors to represent target doors. 
    
    def compute_optimal_velocity(self, t, m):
        '''
        Solve the HJB equation and compute the optimal velocities.

        Returns
        -------
        None.

        '''
        
        nx = self.Nx
        ny = self.Ny
        dx = self.dx
        dy = self.dy
        
        self.t = t
        self.nt_opt = round((self.T - self.t)/self.dt)
        
        nt = self.nt_opt
     
        def hjb(t, phi, m):
        
            phi_temp = np.empty((ny+2,nx+2))
            phi_temp[1:-1,1:-1] = phi.reshape(ny,nx).copy()
            
            phi_temp[0,:] = phi_temp[2,:] 
            phi_temp[-1,:] = phi_temp[-3,:]
            phi_temp[:,-1] =  phi_temp[:,-3] 
            phi_temp[:,0]  =  phi_temp[:,2]  
            
            lap = (phi_temp[:-2,1:-1] + phi_temp[2:,1:-1] + \
                              phi_temp[1:-1,:-2] + phi_temp[1:-1,2:] - \
                              4*phi_temp[1:-1,1:-1])/(dx*dy)
             
            
            phi_temp[1:-1,1:-1] = -0.5*self.sigma**2*lap -\
                ((self.V + self.g*m)*phi_temp[1:-1,1:-1])/(self.mu*self.sigma**2)
                
            phi_temp[1:-1,1:-1][self.V<0] = 0
         
            return phi_temp[1:-1,1:-1].reshape(nx*ny)
        
        # We compute the optimal velocity using the Cole-Hopf version of the value function u
        
        def vels(phi,mu):
            
            phi_temp = phi.reshape(ny,nx).copy()
            
            phi_temp = phi_temp*(phi_temp > self.lim) + self.lim*(phi_temp < self.lim)
            
            grad_x = (phi_temp[1:-1,2:] - phi_temp[1:-1,:-2])/(2*dx)
            grad_y = (phi_temp[2:,1:-1] - phi_temp[:-2,1:-1])/(2*dy)
            
            phi_temp = phi_temp[1:-1,1:-1]*(phi_temp[1:-1,1:-1] > self.lim) + self.lim*(phi_temp[1:-1,1:-1]<self.lim)
        
            vx = grad_x/(mu*phi_temp)
            vy = grad_y/(mu*phi_temp)
            
            norm = np.sqrt(vx**2+vy**2)
            
            den = norm*(norm > self.lim) +  (norm < self.lim)
            
            return (vx*(norm > self.lim))/den, (vy*(norm > self.lim))/den
            
        
        phi_T = self.phi_T
      
        # We choose the integration time span and the time discretization
       
        t_span = (self.T,0)
        t_events = np.linspace(self.T,0,self.nt_opt)

        sol = solve_ivp(hjb, t_span, phi_T, method ='RK45',t_eval = t_events, args = (m,))
        
        # We create the floor field prescribing agents velocity 
        
        for i in np.arange(nt-1,0,-1):
            
            vx,vy = vels(sol.y[:,nt - i ],self.mu)
            self.vx_opt[i-1] = vx
            self.vy_opt[i-1] = vy

        print('Optimal trajectories have been learnt for ' + self.target )
    
    # Given an agent's position, the 'choose_optimal_velocity' method
    # assings the corresponding optimal velocity at a given time 
    # by choosing the grid point correspoding to the floored position coordinates
    
    def choose_optimal_velocity(self,pos,t):
        '''
        Given the agent position, interpolates the optimal velocity at time t. 

        Parameters
        ----------
        pos : numpy.array()
            Coordinates of the agent's position.
        t : float
            Time.

        Returns
        -------
        numpy.array()
            Optimal velocity coordinates.

        '''
        x,y = pos
        
        if t >= self.nt_opt-1:
            return np.array((0.,0.),dtype = float)
        else: 
            if x < self.room_length-self.dx:
                j = int(x//self.dx)
                if x > self.dx:
                    j = [j,j+1]
            else:
                j = self.Nx - 3
            if y < self.room_height-self.dy:
                i = int(y//self.dy)
                if y > self.dy:
                    i = [i,i+1]
            else:
                i = self.Ny - 3
    
            vx = np.mean(self.vx_opt[t][i,j])
            vy = np.mean(self.vy_opt[t][i,j])
            
            return np.array((vx ,vy), dtype = float)