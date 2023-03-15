import matplotlib
import pandas
#matplotlib.use('Agg')
#from toolbox import *
from scipy.spatial import Voronoi, Delaunay
from matplotlib.patches import Polygon
from scipy.optimize import linprog
from shapely.geometry import Polygon as polyg

from scipy.spatial import ConvexHull
import sys, os, os.path

params = {'backend': 'pdf',
		  'axes.labelsize': 22,
		  'font.size': 15,
		  'xtick.labelsize': 15,
		  'legend.fontsize':14,
		  'ytick.labelsize': 15,
		  'text.usetex': True,
		  'figure.figsize': (7,5)}
plt.rcParams.update(params)

### Parameters
r_cg= 0.1 # coarse-graining size
L_max= 8
nb_cg= int(L_max/r_cg)+1
radius_obs= 0.37


epsilon= 0.002 # small number
DoPlot= True
IsObstacle= True
### ###

	
individual_areas= list([])

def getNextMinus1(liste_X, liste_Y, current_index):
	nb_lines= len(liste_X)
	for cpt in range(current_index, nb_lines):
		if liste_X[cpt]== -1 and liste_Y[cpt]== -1:
			return cpt
	return nb_lines
	
def getNextNonMinus1(liste_X, liste_Y, current_index):
	nb_lines= len(liste_X)
	for cpt in range(current_index, nb_lines):
		if liste_X[cpt]!= -1 or liste_Y[cpt]!= -1:
			return cpt
	return -1
	
def PolyArea(x,y):
	return 0.5*abs(dot(x,roll(y,1))-dot(y,roll(x,1)))
	
	

def in_hull(points, x):
	n_points = len(points)
	c = zeros(n_points)
	A = r_[points.T,ones((1,n_points))]
	b = r_[x, ones(1)]
	lp = linprog(c, A_eq=A, b_eq=b,method="revised simplex", options={'presolve':False, 'tol': 1e-11})
	return lp.success
	
	
def in_nc_polygon(points,r): # test if r is inside non-convex polygon by drawing line from r to infinity and counting intersections
	r_infty= (101.676,100.) # point at infinity
	segment1= (r, r_infty)
	
	
	nb_inter= 0 # number of intersections
	
	
	for cpt in range(len(points)):
		cptp1= (cpt+1) % len(points)
		segment2= (points[cpt], points[cptp1])
		if segment_intersection(segment1, segment2)!=-1:
			nb_inter+= 1
		
	return (nb_inter%2 == 1)
	


colors = list(matplotlib.colors.CSS4_COLORS)





cm=plt.get_cmap("gist_heat")


def get_r_mesh(i,j): 
	return ( -0.5*L_max+float(i)*r_cg, -0.5*L_max+float(j)*r_cg )
	

def getActiveTraj(myfolder, mytime_int): 
	# returns trajectories that may have been active between mytime_int and mytime_int+1
	liste= []
	
	for myTraj in os.listdir(myfolder):
		with open(myfolder+"/"+myTraj,'r') as myfile:
			for i, line in enumerate(myfile):
				myline= line.split()
				time_loc= float(myline[0])
				if i==0 and time_loc>mytime_int+1:
					break
				if time_loc>mytime_int:
					liste.append(myTraj)
					break
		
	return liste
	
	

	
	
def get_xy():
	xy= list([])
	for x in arange(-radius_obs,radius_obs,0.03):
		xy.append( (x, sqrt(radius_obs**2-x**2) ) )
	for x in arange(radius_obs,-radius_obs,-0.03):
		xy.append( (x, -sqrt(radius_obs**2-x**2) ) )
	return array(xy)


"""
def PlotPos(myfolder, obstacle_id, mytime, activeTrajs, area_array):
	
	# Get obstacle (or crossing pedestrian)'s position
	
	r_obstacle= getPosAtTime(myfolder, str(obstacle_id),mytime)
	
	if r_obstacle==-1 or distance(r_obstacle,[-1,-1])<0.01: # if obstacle not detected at that time
		return
				
	positions= []
	positions.append( (0.,0.) ) # obstacle position, so that obstacle has index 0
	displ= []
	displ.append( (0.,0.) )
	
	for myTraj in activeTrajs:
		if myTraj==str(obstacle_id):
			continue
		
		dt=0.8
		r= getPosAtTime(myfolder,myTraj,mytime)
		r_nexttime= getPosAtTime(myfolder,myTraj,mytime+dt)
		if r==-1 or distance(r,[-1,-1])<0.01:
			continue
		
		positions.append( subtract(r,r_obstacle) )
		
		if r_nexttime!=-1:
			displ.append( subtract(r_nexttime,r) )
		else:
			displ.append( (-0.005*dt,-0.005*dt) )
	
	# check if obstacle is surrounded by pedestrians to know if it is worth including it
	Left= sum(list(map(lambda r: bool(r[0]<-0.5), positions)))
	Right= sum(list(map(lambda r: bool(r[0]>0.5), positions)))
	Top= sum(list(map(lambda r: bool(r[1]<-0.5), positions)))
	Bottom= sum(list(map(lambda r: bool(r[1]>0.5), positions)))
	IsSurrounded= bool( (Left>=1) and (Right>=1) and (Top>=1) and (Bottom>=1) )
	# WRITE POSITIONS IN FILE
	with open("/home/alexandre/Bureau/Orsay/Python/PedExp/Positions_rho=2.5/%spositions_%i_%.1f.csv"%("" if IsSurrounded else "z_", obstacle_id,mytime),'w') as monfichier:
		for cpt in range(len(positions)):
			pos= positions[cpt]
			dis= displ[cpt]
			monfichier.write("%.3f %.3f %.3f %.3f\n"%(pos[0],pos[1], dis[0]/dt, dis[1]/dt))
	
	
		
	return
"""

rho= "4"
os.chdir("/home/alexandre/Bureau/Orsay/Python/PedExp/Positions_rho="+rho)
#os.chdir("/home/alexandre/Bureau/Orsay/Python/PedExp/RandomPositions_rho="+rho)
#os.chdir("/home/alexandre/Bureau/Orsay/Python/PedExp/BackPositions")

N=100 # number of gridpoints on one axis of the matrix
L=3. # linear size of the matrix in meters
sigma_Gauss= 0.185
cutoff= 3.0 # in units of sigma_Gauss

def pos_from_ij(i,j):
	return (L*float(i-N//2)/float(N), L*float(j-N//2)/float(N))
	
def gaussian_trunc(x,y): # gaussian truncated at R= 3 * sigma_Gauss
	R2= (x*x+y*y)
	if R2> (cutoff*sigma_Gauss)**2:
		return 0.0
	return exp( - 0.5 * R2 / sigma_Gauss**2 ) / (2.0 * pi * sigma_Gauss**2 * (1.0 - exp(-0.5*cutoff*cutoff)) )

Sum_of_rho= zeros((N,N), dtype='d')
Times_in_hull= zeros((N,N), dtype='i')

X= zeros((N,N), dtype='d')
Y= zeros((N,N), dtype='d')

for i in range(N):
	for j in range(N):
		xy= pos_from_ij(i,j)
		X[i,j]= xy[0]-0.5*L/float(N)
		Y[i,j]= xy[1]-0.5*L/float(N)
		

for fichier in os.listdir("."):
	if fichier[0:3]!="pos":
		continue
	
	print("* Analysing file ", fichier)
	
	m= re.match("positions_(\d+)_(\d+.\d+).csv", fichier)
	obstacle_ID= int(m.group(1))
	mytime= float(m.group(2))
	data= loadtxt(fichier, delimiter=" ", skiprows=1) # skiprows instruction in order NOT to count the obstacle (first line)
	
	positions= array([ (datum[0],datum[1]) for datum in data])
	#print(ConvexHull(array(positions)).vertices)
	hull_ext = positions[list(ConvexHull(array(positions)).vertices)]
	
	
	for i in range(N):
		for j in range(N):
			if not in_hull(hull_ext, pos_from_ij(i,j)):
				continue
			
			Times_in_hull[i,j]+=1
			
			for pos in positions:
				#plt.plot(pos[0], pos[1], 'bo', markersize=1)
				Sum_of_rho[i,j]+= gaussian_trunc( pos[0]-X[i,j], pos[1]-Y[i,j] )
				#plt.plot(xy[0],xy[1],'o', color= 'green' if InHull else 'red')
	
	#plt.pcolormesh(X,Y,Sum_of_rho)
	
		
	#for pos in hull_ext:
	#	plt.plot(pos[0], pos[1], 'rs')



for i in range(N):
	for j in range(N):
		if Times_in_hull[i,j]>0:
			Sum_of_rho[i,j]/= float(Times_in_hull[i,j])


with open("density_field_"+rho,'w') as monfichier:
	monfichier.write("x y rho")
	for i in range(N):
		for j in range(N):
			monfichier.write("\n%.2f %.2f %.3f"%(X[i,j],Y[i,j],Sum_of_rho[i,j]))

print("Done")
"""
plt.pcolormesh(X,Y,Sum_of_rho)
plt.colorbar()
plt.show()
"""
