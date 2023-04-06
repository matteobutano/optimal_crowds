from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import json

# Create class to describe simulation

class simulation:
    def __init__(self, mode,T = 10):
        with open('optimal_crowds/config.json') as f:
            var = json.loads(f.read())
        
        # Read doors
        self.doors = np.empty((len(var['doors']),5))
        for i,door in enumerate(var['doors']):
            self.doors[i,:] = np.array(var['doors'][str(door)])
        
        # Read room 
        self.room_length = var['room_length']
        self.room_height = var['room_height']
        self.pot = var['hjb_params']['potential']
        
        self.Nx = var['Nx']
        self.Ny = var['Ny']
        self.sigma_convolution = var['sigma_convolution']
        
             
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
        
        for walls in var['walls']:
            wall = var['walls'][walls]

            mask_X = abs(self.X_opt-wall[0]) < wall[2]/2
            mask_Y = abs(self.Y_opt-wall[1]) < wall[3]/2
           
            
            V_temp = np.zeros((self.Ny,self.Nx))
            
            V_temp[mask_X*mask_Y] = self.pot
            
            self.V += V_temp
         
        for holes in var['holes']:
            hole = var['holes'][holes]
        
            hole_X = abs(self.X_opt-hole[0]) < hole[2]/2
            hole_Y = abs(self.Y_opt-hole[1]) < hole[3]/2
            
            self.V[hole_X*hole_Y] = 0   
        
        for cyls in var['cylinders']:
            cyl = var['cylinders'][cyls]
            
            V_temp =  np.zeros((self.Ny,self.Nx))
            
            V_temp[np.sqrt((self.X_opt-cyl[0])**2 + (self.Y_opt-cyl[1])**2) < cyl[2]] = self.pot
            
            self.V+= V_temp
        
        self.V = self.pot *(self.V <= self.pot)  
        
        for door in var['doors']:
            door = var['doors'][door]
            if door[2] > 0:
                door_X = abs(self.X_opt - door[0]) < door[2]/2
                door_Y = abs(self.Y_opt - door[1]) < 0.2*door[2]
            if door[3] > 0:
                door_Y = abs(self.Y_opt - door[1]) < door[3]/2
                door_X = abs(self.X_opt - door[0]) < 0.2*door[3]
            self.evacuator[door_X*door_Y] = 0
            self.V[door_X*door_Y] = door[4]
            self.phi_0[door_X*door_Y] = door[4]
            
        self.phi_0 = self.phi_0.reshape(self.Nx*self.Ny)
        
        # Init time variables 
        self.time = 0.
        self.simu_step = 0
        self.dt = var['dt']
        
        # Init population
        self.N = var['N']
        self.inside = var['N']
         
        # Init pedestrians features
        self.relaxation = var['relaxation']
        self.rep_radius = var['repulsion_radius']
        self.rep_int = var['repulsion_intensity']
        self.noise_intensity = var['noise_intensity']
        self.des_v = var['des_v']
        self.initial_box = var['initial_box']
        self.agents = np.empty(self.N,dtype=object)
        self.initial_density = var['hjb_params']['initial_density']
        self.m_0 = np.zeros((self.Ny,self.Nx),dtype = float) 
            
        self.type = mode
        
        self.optimal = optimal_trajectories(T)
        
        if self.type == 'abm':
           # Create crowd
           xs = np.random.uniform(self.initial_box[0], self.initial_box[1],self.N)
           ys = np.random.uniform(self.initial_box[2],self.initial_box[3],self.N)
           for i in range(self.N):
               self.agents[i] = ped(xs[i], ys[i], 0, 0, self.doors, self.room_length, self.room_height)
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
        
        if self.type == 'abm':
        
            if mode == 'scatter':
                scat_x = [self.agents[i].position()[0] for i in range(self.N) if self.agents[i].status]
                scat_y = [self.agents[i].position()[1] for i in range(self.N) if self.agents[i].status]
                plt.scatter(scat_x,scat_y,color = 'blue')
             
            if mode == 'arrows':
                if self.inside > 0:
                    scat_x = [self.agents[i].position()[0] for i in range(self.N) if self.agents[i].status]
                    scat_y = [self.agents[i].position()[1] for i in range(self.N) if self.agents[i].status]
                    scat_vx = [self.agents[i].velocity()[0] for i in range(self.N) if self.agents[i].status]
                    scat_vy = [self.agents[i].velocity()[1] for i in range(self.N) if self.agents[i].status]
                    plt.quiver(scat_x,scat_y,scat_vx, scat_vy,color = 'blue')
                else:
                    plt.plot()                 
            
            if mode == 'density':
                X,Y,d = self.gaussian_density(self.sigma_convolution, self.Nx, self.Ny)
                plt.pcolor(X,Y,d.reshape(self.Ny,self.Nx))
                plt.colorbar()
            
        elif self.type == 'mfg':
            
            plt.pcolor(self.X_opt, self.Y_opt,self.m_0 + 2*self.initial_density*self.V/self.pot)
                    
        plt.imshow(np.flip(self.V,axis = 0),extent=[0,self.room_length,0,self.room_height])
        plt.xlim([0,self.room_length])
        plt.ylim([0,self.room_height])
        title = 't = {:.2f}s exit = {}/{}'.format(self.time,self.N - self.inside,self.N)
        plt.title(title)
        plt.show()
            
    def step(self,dt,verbose =  False):
        for i in  np.random.choice(np.arange(self.N),self.N,replace=False):
            # Summon agent
            agent = self.agents[i]
    
            # Check if agent has reached the a door
            if agent.status:
            
                # Compute desired velocity 
                if self.simu_step < self.optimal.nt_opt:
                    vx_opt,vy_opt = self.optimal.choose_optimal_velocity(agent.position(), self.simu_step)
                    des_x, des_y = self.des_v*np.array((vx_opt,vy_opt),dtype = float) 
                
                else:
                    des_x,des_y= agent.desired_velocity(self.des_v)
                
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

    def run(self, verbose = True, draw = False, mode = 'scatter'):
        
        if self.type == 'abm':
        
            while self.inside > 0:    
                self.step(self.dt,verbose = verbose)
                if draw:
                    self.draw(mode)
                    plt.show()
                
            print('Evacuation complete!')
            
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
        return X,Y,d.reshape(self.Nx*self.Ny)
        
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
        

class optimal_trajectories:
    def __init__(self,T):
        
        with open('optimal_crowds/config.json') as f:
            var = json.loads(f.read())
            
        # Read room 
        self.room_length = var['room_length']
        self.room_height = var['room_height']
        self.doors = np.empty((len(var['doors']),5))
        for i,door in enumerate(var['doors']):
            self.doors[i,:] = np.array(var['doors'][str(door)])
        
        # Read optimization parameters
        self.Nx = var['Nx']
        self.Ny = var['Ny']
        self.g = var['hjb_params']['g']
        self.sigma = var['hjb_params']['sigma']
        self.mu = var['hjb_params']['mu']
        self.pot = var['hjb_params']['potential']
        
        
        self.dt = var['dt']
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
        self.lim = 10e-6
        
        for walls in var['walls']:
            wall = var['walls'][walls]

            mask_X = abs(self.X_opt-wall[0]) < wall[2]/2
            mask_Y = abs(self.Y_opt-wall[1]) < wall[3]/2
           
            
            V_temp = np.zeros((self.Ny,self.Nx))
            
            V_temp[mask_X*mask_Y] = self.pot
            
            self.V += V_temp
         
        for holes in var['holes']:
            hole = var['holes'][holes]
        
            hole_X = abs(self.X_opt-hole[0]) < hole[2]/2
            hole_Y = abs(self.Y_opt-hole[1]) < hole[3]/2
            
            self.V[hole_X*hole_Y] = 0   
        
        for cyls in var['cylinders']:
            cyl = var['cylinders'][cyls]
            
            V_temp =  np.zeros((self.Ny,self.Nx))
            
            V_temp[np.sqrt((self.X_opt-cyl[0])**2 + (self.Y_opt-cyl[1])**2) < cyl[2]] = self.pot
            
            self.V+= V_temp
        
        self.V = self.pot *(self.V <= self.pot)  
        
        for door in var['doors']:
            door = var['doors'][door]
            if door[2] > 0:
                door_X = abs(self.X_opt - door[0]) < door[2]/2
                door_Y = abs(self.Y_opt - door[1]) < 0.2*door[2]
            if door[3] > 0:
                door_Y = abs(self.Y_opt - door[1]) < door[3]/2
                door_X = abs(self.X_opt - door[0]) < 0.2*door[3]
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
            
        epoch = 0
        
        err = 1
        
        while err > 10e-4:
            
            sol_phi = solve_ivp(phi, t_span_phi, self.phi_0, 
                                method ='RK45',t_eval = t_events_phi, args =(m_total,dt))
            sol_gam = solve_ivp(gam, t_span_gam, m_0.reshape(nx*ny)/sol_phi.y[:,-1], 
                                method ='RK45',t_eval = t_events_gam, args =(m_total,dt))
            
            phi_total = np.flip(sol_phi.y.reshape((ny,nx,nt)),axis = 2)
            gam_total =  sol_gam.y.reshape((ny,nx,nt))
            
            err = np.mean([(phi_total[:,:,i]*gam_total[:,:,i]-\
                                   m_total[:,:,i])**2 for i in range(nt)])
            if verbose:
                print('Epoch {}, error = {:.4f}'.format(epoch,err))
            
            m_total = 0.1*phi_total*gam_total + 0.9*m_total
            
            epoch+=1


        m_total = phi_total*gam_total 
        
        if draw: 
            
            for t in range(nt):
                
                plt.pcolor(self.X_opt,self.Y_opt,m_total[:,:,t]*self.evacuator)
                plt.title(t)        
                plt.clim([0,2])
                plt.colorbar()
                
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
            
            return vx ,vy




        
            

        
        
        
            
            
