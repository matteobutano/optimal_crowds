import evacuation as evac
import numpy as np

N = 37
dt = 0.01

simu = evac.simulation(N, dt)

simu.run()
