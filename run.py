import evacuation as evac
import numpy as np
import sys

sim_number = str(sys.argv[1])

N = 300
dt = 0.01

simu = evac.simulation(N, dt)

simu.run()

xs,ys = simu.initial_positions()

times = simu.evac_times()

np.savetxt('data_abm_evac/init_pos_times_N=' + str(N) + '_dt=' + str(dt) + '/init_pos_time-' + sim_number + '.txt',np.vstack([xs,ys,times]))
print('Data saved!')
