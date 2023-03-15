import numpy as np
import matplotlib.pyplot as plt


# Room Description
room_length = 6
room_height = 4
door_width = 0.5
door_position = room_length / 2

# Discretization

Nx = 150
Ny = 100
dx = room_length/(Nx-1)
dy = room_height/(Ny-1)

X,Y = np.meshgrid(np.linspace(0,room_length,Nx+1), np.linspace(0,room_height,Ny+1))

X = X[:-1,:-1] + dx/2
Y = np.flip(Y[:-1,:-1] + dy/2,axis = 0)

agents_time = np.load('../data_abm_evac/positions-3.npy',allow_pickle= True)

def gaussian(x,y,x_agent,y_agent,sigma):
    c_x = x - x_agent
    c_y = y - y_agent
    C = np.sqrt(4*np.pi**2*sigma**2)
    return np.exp(-(c_x**2 + c_y**2)/(2*sigma**2))/C

for t in range(agents_time.shape[0]):
    agents = np.array(agents_time[t])
    density = np.zeros((Ny,Nx))
    for agent in agents:
        density += gaussian(X, Y, agent[0], agent[1], 0.2)
    
    plt.pcolor(X,Y,density)
    plt.colorbar()
    plt.show()

