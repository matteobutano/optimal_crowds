from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json

# Create class that computes optimal trajs and mfg
class optimal_trajectories:
    def __init__(self,room,T):
        
        with open('optimal_crowds/config.json') as f:
            var_config = json.loads(f.read())
            
        with open('rooms/'+room+'.json') as f:
            var_room = json.loads(f.read())
            
        # Read room 
        self.room_length = var_room['room_length']
        self.room_height = var_room['room_height']
        self.doors = np.empty((len(var_room['doors']),5))
        for i,door in enumerate(var_room['doors']):
            self.doors[i,:] = np.array(var_room['doors'][str(door)])
        
        # Read optimization parameters
        self.Nx = var_room['Nx']
        self.Ny = var_room['Ny']
        self.g = var_config['hjb_params']['g']
        self.sigma = var_config['hjb_params']['sigma']
        self.mu = var_config['hjb_params']['mu']
        self.pot = var_config['hjb_params']['potential']
        
        
        self.dt = var_room['dt']
        self.T = T
        self.nt_opt = round(self.T/self.dt)
            
        # Create boundary conditions, doors and final cost
        
        self.dx = self.room_length/(self.Nx-1)
        self.dy = self.room_height/(self.Ny-1)

        self.X_opt, self.Y_opt = np.meshgrid(np.linspace(0,self.room_length,self.Nx)
                                             ,np.linspace(0,self.room_height,self.Ny))
        
        self.vx_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        self.vy_opt = np.empty((self.nt_opt-1,self.Ny-2,self.Nx-2))
        
        self.u_0 = np.zeros((self.Ny,self.Nx),dtype = float)
        self.phi_0 =  np.zeros((self.Ny,self.Nx),dtype = float)
        self.evacuator = np.zeros((self.Ny,self.Nx),dtype = float) + 1
        
        self.V = np.zeros((self.Ny,self.Nx)) + self.pot
        self.V[1:-1,1:-1] = 0
        self.lim = 10e-10
        
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
            self.phi_0[door_X*door_Y] = door[4]
            
        self.phi_0 = self.phi_0.reshape(self.Nx*self.Ny)
        
       
    def draw(self,mode):
        if mode == 'trajectories':
            for i in range(self.nt_opt-1):
                if i < self.nt_opt-2:
                    plt.quiver(self.X_opt[1:-1,1:-1],self.Y_opt[1:-1,1:-1],self.vx_opt[i],self.vy_opt[i])
                else:
                    plt.plot()
                for door in self.doors:
                    plt.plot([door[0]-door[2]/2, door[0] + door[2]/2],[door[1] - door[3]/2, door[1] + door[3]/2], 'r-', linewidth=4)
                plt.xlim([0,self.room_length])
                plt.ylim([0,self.room_height])
                title = 't = {:.2f}s'.format(i*self.dt)
                plt.title(title)
                plt.show()
        elif mode == 'setup':
            plt.pcolor(self.X_opt,self.Y_opt,self.V)
            plt.colorbar()
            plt.show()
        
            
    def mean_field_game(self,m_0,draw, verbose = False):
        
        nx = self.Nx
        ny = self.Ny
        dx = self.dx
        dy = self.dy
        T = self.T
        dt = self.dt
        nt = int( T//dt + 1)
    
        def phi(t,phi,m,dt):
            
            i = int(np.round(t,2)//dt)
            
            m_temp = m[:,:,i]*self.evacuator
            
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
                ((self.V+self.g*m_temp)*phi_temp[1:-1,1:-1])/(self.mu*self.sigma**2) 
            
            return phi_temp[1:-1,1:-1].reshape(nx*ny)

        def gam(t,gam,m,dt):
            
            i = int(np.round(t,2)//dt)
            
            m_temp = m[:,:,i]*self.evacuator
            
            gam_temp = np.empty((ny+2,nx+2))
            gam_temp[1:-1,1:-1] = gam.reshape(ny,nx).copy()
            
            gam_temp[0,:] = gam_temp[2,:] 
            gam_temp[-1,:] = gam_temp[-3,:]
            gam_temp[:,-1] =  gam_temp[:,-3] 
            gam_temp[:,0]  =  gam_temp[:,2]  
            
            lap = (gam_temp[:-2,1:-1] + gam_temp[2:,1:-1] + gam_temp[1:-1,:-2] +\
                   gam_temp[1:-1,2:] -4*gam_temp[1:-1,1:-1])/(dx*dy)
                
            gam_temp[1:-1,1:-1] = 0.5*self.sigma**2*lap +\
                ((self.V + self.g*m_temp)*gam_temp[1:-1,1:-1])/(self.mu*self.sigma**2) 
                        
            return gam_temp[1:-1,1:-1].reshape(nx*ny)
        
        print('Starting MFG!')
        
        t_span_phi = (T,0)
        t_span_gam = (0,T)
        
        t_events_phi = np.linspace(T,0,nt)
        t_events_gam = np.linspace(0,T,nt)

        m_0_total = np.zeros((ny,nx,nt))
        m_0_total[:,:,0] = m_0

        sol_phi = solve_ivp(phi, t_span_phi, self.phi_0,
                            method ='RK45',t_eval = t_events_phi, args =(m_0_total,dt))
        sol_gam = solve_ivp(gam, t_span_gam, m_0.reshape(nx*ny)/sol_phi.y[:,-1],
                            method ='RK45',t_eval = t_events_gam, args =(m_0_total,dt))

        phi_total = np.flip(sol_phi.y.reshape((ny,nx,nt)),axis = 2)
        gam_total =  sol_gam.y.reshape((ny,nx,nt))
        m_total = phi_total*gam_total
            
        if draw:
            
            for t in range(nt):
                
                plt.figure(figsize = (self.room_length,self.room_height))
                plt.imshow(np.flip(m_total[:,:,t]*self.evacuator + self.V/self.pot,axis = 0),extent=[0,self.room_length,0,self.room_height])
                plt.clim([0,5])
                plt.colorbar()
                plt.title('Optimal evacuation, t = {:.2f}s'.format(dt*t))
                plt.show()
        
        epoch = 0
        
        err = 10e6
        
        early_stop = 0
        
        while (err > 10e-4) & (early_stop < 5):
            
            sol_phi = solve_ivp(phi, t_span_phi, self.phi_0, 
                                method ='RK45',t_eval = t_events_phi, args =(m_total,dt))
            sol_gam = solve_ivp(gam, t_span_gam, m_0.reshape(nx*ny)/sol_phi.y[:,-1], 
                                method ='RK45',t_eval = t_events_gam, args =(m_total,dt))
            
            phi_total = np.flip(sol_phi.y.reshape((ny,nx,nt)),axis = 2)
            gam_total =  sol_gam.y.reshape((ny,nx,nt))
            
            new_err = np.mean([(phi_total[:,:,i]*gam_total[:,:,i]-\
                                   m_total[:,:,i])**2 for i in range(nt)])
                
            if new_err >= err:
                early_stop+=1
                
            else:
                early_stop = 0
                    
            err = new_err 
            
            if verbose:
                print('Epoch {}, error = {:.4f}'.format(epoch,err))
            
            m_total = 0.1*phi_total*gam_total + 0.9*m_total
            
            epoch+=1


        m_total = phi_total*gam_total 
        
        if draw: 
            
            for t in range(nt):
                
                plt.figure(figsize = (self.room_length,self.room_height))
                plt.imshow(np.flip(m_total[:,:,t]*self.evacuator + self.V/self.pot,axis = 0),extent=[0,self.room_length,0,self.room_height])
                plt.title(t)
                plt.clim([0,5])
                plt.colorbar()
                plt.title('Nash equilibrium, t = {:.2f}s'.format(dt*t))
                
                plt.show()
            
        return m_total
    
    def compute_optimal_velocity(self):
        
        nx = self.Nx
        ny = self.Ny
        dx = self.dx
        dy = self.dy
        nt = self.nt_opt
     
        def hjb(t,phi,i):
        
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
         
            return phi_temp[1:-1,1:-1].reshape(nx*ny)
        
        def vels(phi,mu):
            
            lim = self.lim
            
            phi_temp = phi.reshape(ny,nx).copy()
            
            grad_x = (phi_temp[1:-1,2:] - phi_temp[1:-1,:-2])/(2*dx)
            grad_y = (phi_temp[2:,1:-1] - phi_temp[:-2,1:-1])/(2*dy)
            
            phi_den = phi_temp[1:-1,1:-1]*(phi_temp[1:-1,1:-1] > lim) + lim*(phi_temp[1:-1,1:-1]<lim)
            
            vx = grad_x/(mu*phi_den)
            vy = grad_y/(mu*phi_den)
            
            norm = np.sqrt(vx**2 + vy**2)
            
            norm = norm*(norm > lim) + lim*(norm < lim)
            
            return vx/norm,vy/norm
        
        phi_0 = self.phi_0
      
        t_span = (self.T,0)
      
        t_events = np.linspace(self.T,0,self.nt_opt)

        sol = solve_ivp(hjb, t_span, phi_0, method ='RK45',t_eval = t_events,args = (0,))
        
        for i in np.arange(nt-1,0,-1):
            
            vx,vy = vels(sol.y[:,nt - i ],self.mu)
            self.vx_opt[i-1] = vx
            self.vy_opt[i-1] = vy

        print('Optimal trajectories have been learnt!')
    
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