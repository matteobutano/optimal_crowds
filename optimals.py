# author Matteo Butano
# email: matteo.butano@universite-paris-saclay.fr
# institution: CNRS, Université Paris-Saclay, LPTMS

# Modules are imported 

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json

# The 'optimals' class is used to solve the HJB equation giving the optimal
# trajectories as per the cost functional cited in my pubblications.

class optimals:
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
       
        # Time discretization 
        
        self.dt = var_config['dt']
        self.T = T
        self.nt_opt = round(self.T/self.dt)
            
        # Create boundary conditions and final cost
         
        self.vx_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        self.vy_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        
        self.phi_T =  np.zeros((self.Ny,self.Nx),dtype = float) + 1
        
        # Create potential V
        
        self.V = np.zeros((self.Ny,self.Nx)) + self.pot
        self.V[1:-1,1:-1] = 0
        self.lim = 10e-10
        
        # Read room 
        
        self.doors = np.empty((len(var_room['doors']),4))
        for i,door in enumerate(var_room['doors']):
            self.doors[i,:] = np.array(var_room['doors'][str(door)])
        
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
            
            self.V[door_X*door_Y] = var_config['hjb_params']['door_potential']
            
        self.phi_T = self.phi_T.reshape(self.Nx*self.Ny)
        
    # The 'draw_optimal_velocities' method draws the velocities obtained 
    # by solving the HJB equation in the Cole-Hopf transformation. 
       
    def draw_optimal_velocity(self):
        
        for i in range(self.nt_opt-1):
            
            if i < self.nt_opt-2:
                plt.quiver(self.X_opt[1:-1,1:-1],self.Y_opt[1:-1,1:-1],self.vx_opt[i],self.vy_opt[i])
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
    
    def compute_optimal_velocity(self):
        
        nx = self.Nx
        ny = self.Ny
        dx = self.dx
        dy = self.dy
        nt = self.nt_opt
     
        def hjb(t,phi):
        
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
                ((self.V)*phi_temp[1:-1,1:-1])/(self.mu*self.sigma**2)
                
            phi_temp[1:-1,1:-1][self.V<0] = 0
         
            return phi_temp[1:-1,1:-1].reshape(nx*ny)
        
        # We compute the optimal velocity using the Cole-Hopf version of the value function u
        
        def vels(phi,mu):
            
            lim = self.lim
            
            phi_temp = phi.reshape(ny,nx).copy()
            
            grad_x = (phi_temp[1:-1,2:] - phi_temp[1:-1,:-2])/(2*dx)
            grad_y = (phi_temp[2:,1:-1] - phi_temp[:-2,1:-1])/(2*dy)
            
            phi_den = phi_temp[1:-1,1:-1]*(phi_temp[1:-1,1:-1] > lim) + lim*(phi_temp[1:-1,1:-1]<lim)
            
            vx = grad_x/(mu*phi_den)
            vy = grad_y/(mu*phi_den)
            
            norm = np.sqrt(vx**2+vy**2)
            
            norm= norm*(norm > lim) + lim*(norm < lim)
        
            return vx/norm,vy/norm
        
        phi_T = self.phi_T
      
        # We choose the integration time span and the time discretization
       
        t_span = (self.T,0)
        t_events = np.linspace(self.T,0,self.nt_opt)

        sol = solve_ivp(hjb, t_span, phi_T, method ='RK45',t_eval = t_events)
        
        # We create the floor field prescribing agents velocity 
        
        for i in np.arange(nt-1,0,-1):
            
            vx,vy = vels(sol.y[:,nt - i ],self.mu)
            self.vx_opt[i-1] = vx
            self.vy_opt[i-1] = vy

        print('Optimal trajectories have been learnt!')
    
    # Given an agent's position, the 'choose_optimal_velocity' method
    # assings the corresponding optimal velocity at a given time 
    # by choosing the grid point correspoding to the floored position coordinates
    
    def choose_optimal_velocity(self,pos,t):
        x,y = pos
        if t >= self.nt_opt-1:
            return (0.,0.)
        else: 
            if x < self.room_length-self.dx:
                j = int(x//self.dx)
            else:
                j = self.Nx - 3
            if y < self.room_height-self.dy:
                i = int(y//self.dy)
            else:
                i = self.Ny - 3
    
            vx = self.vx_opt[t][i,j]
            vy = self.vy_opt[t][i,j]
            
            return np.array((vx ,vy), dtype = float)