import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#sym_number = float(sys.argv[1])
sym_number = 1

# Pedestrians Description
num_pedestrians = 500
init_num_pedestrians = num_pedestrians
repulsion_radius = 0.2
repulsion_intensity = -2
noise_intensity = 0.1

# Room Description
room_length = 6
room_height = 4
door_width = 0.5
door_position = room_length / 2


# Time Discretization
time_step = 0.01
t = 0

# Initialize pedestrian positions and velocities
pedestrian_positions = []
pedestrian_velocities = []
positions = []
velocities = []
for i in range(num_pedestrians):
    x = random.uniform(repulsion_radius, room_length-repulsion_radius)
    y = random.uniform(repulsion_radius, room_height-repulsion_radius)
    pedestrian_positions.append((x, y))
    pedestrian_velocities.append((0, 0))

# Graphics 
fig,axs = plt.subplots(figsize=(8,6))
ims = []

while num_pedestrians > 0:
    completed_pedestrians = []
    for i in range(num_pedestrians):
        # Compute distance to door
        x, y = pedestrian_positions[i]
        distance_to_door = np.sqrt((x-door_position)**2 + y**2)
        
        # Check if pedestrian has reached the door
        if (y < 0.1) and (door_position - door_width/2 < x < door_position + door_width/2):
            completed_pedestrians.append(i)
            continue  # Skip this pedestrian

        # Compute desired velocity 
        des_v = 0.5
        des_x,des_y = (-np.array(pedestrian_positions[i]) + np.array((door_position,0)))/distance_to_door 

        # Compute heading towards door
        if x < door_position:
            heading = 1
        else:
            heading = -1

        # Compute repulsion from nearby pedestrians
        repulsion = np.array((0, 0),dtype = float)
        for j in range(num_pedestrians):
            if j != i:
                dx, dy = -np.array(pedestrian_positions[j]) + np.array(pedestrian_positions[i])
                distance = np.sqrt(dx**2 + dy**2)
                if distance < repulsion_radius:
                    repulsion += -repulsion_intensity * (repulsion_radius - distance) / distance * np.array((dx, dy))
            
        # Compute repulsion from walls
        wall_repulsion = np.array((0, 0),dtype = float)
        if x < repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - x) / repulsion_radius * np.array((1, 0))
        elif x > room_length - repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (room_length - x)) / repulsion_radius * np.array((-1, 0))
        if y < repulsion_radius and (x > door_position + door_width/2 or x < door_position - door_width/2) :
            wall_repulsion += -repulsion_intensity*(repulsion_radius - y) / repulsion_radius * np.array((0, 1))
        elif y > room_height - repulsion_radius:
            wall_repulsion += -repulsion_intensity*(repulsion_radius - (room_height - y)) / repulsion_radius * np.array((0, -1))

        # Compute current velocity with random perturbation and repulsion
        vx, vy = pedestrian_velocities[i]
        current_velocity = (vx + random.uniform(-noise_intensity, noise_intensity), vy + random.uniform(-noise_intensity, noise_intensity)) + repulsion + wall_repulsion 

        # Compute acceleration towards desired velocity
        ax = (des_x*des_v - current_velocity[0]) / 0.5
        ay = (des_y*des_v - current_velocity[1]) / 0.5

        # Update position and velocity
        x += current_velocity[0] * time_step + 0.5 * ax * time_step**2
        y += current_velocity[1] * time_step + 0.5 * ay * time_step**2
        pedestrian_positions[i] = (x, y)
        pedestrian_velocities[i] = (current_velocity[0] + ax * time_step,
                                    current_velocity[1] + ay * time_step)

    # Remove completed pedestrians
    completed_pedestrians.reverse()
    for out in completed_pedestrians:
        pedestrian_positions.pop(out)
        pedestrian_velocities.pop(out)
        num_pedestrians -= 1 
        
    scat_x = [pedestrian_positions[i][0] for i in range(num_pedestrians)]
    scat_y = [pedestrian_positions[i][1] for i in range(num_pedestrians)]
    scat = axs.scatter(scat_x,scat_y,color = 'blue',animated=True)
    axs.plot([door_position-door_width/2, door_position + door_width/2],
            [0, 0], 'r-', linewidth=2)
    axs.set_xlim([0,room_length])
    axs.set_ylim([0,room_height])
    print_time = '{:.1f}'.format(t)
    print_fraction = '{:.1f}'.format(100.-100*num_pedestrians/init_num_pedestrians)
    title = 't = '+ print_time +' s, evac = '+print_fraction+' %'
    print(title + 10*' ',end = '\r')
    title = axs.text(0.5,1.05,title,size=plt.rcParams["axes.titlesize"],ha="center",transform=axs.transAxes)
    ims.append([scat,title]) 
    positions.append(np.array([pedestrian_positions[i] for i in range(num_pedestrians)]))
    velocities.append(np.array([pedestrian_velocities[i] for i in range(num_pedestrians)]))
    t+= time_step

positions = np.array(positions,dtype = object)
velocities = np.array(velocities,dtype = object)
np.save('data_abm_evac/positions-'+str(sym_number), positions)
np.save('data_abm_evac/velocities-'+str(sym_number),velocities)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
ani.save('gifs_abm_evac/simulation-'+str(sym_number)+'.gif', writer='pillow')
