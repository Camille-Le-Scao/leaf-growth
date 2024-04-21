
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from numba import jit

import tables
import h5py   
import pickle

NB_POINTS_0 = 100  #initial points on the front for initial_array()
X_AXIS = list(np.linspace(-50, 50, NB_POINTS_0)) #initial spread on the X axis for initial_array()
D_MIN = 0.2  #a factor for the minimal distance between points
D_MAX = 2  #a factor for the maximal distance between points
MAX_RES = 2 #maximum distance between 2 points of the front to condition new point creation
MIN_RES = 0.5 #minimul between 2 points of the front to condition point suppression

NB_VEINES = 6
NOISE_REDUCTION = 1/100  #noise on the initial points position
D_VEIN = 100/6  #critical distance for vein_propagation()
C = 1
S_GROWTH = D_VEIN / C
A0 = 0  #to tune minimal growth for growth functions
AM = 1 #to tune vein dependant growth for growth function
DT = 0.05
T = 10000000
CENTRAL_WEIGHT = 4  #tune how the growth is more important for the central vein (it's a factor)
BASAL_WEIGHT = 1



### CLASS

class VeinArray:
    
    def __init__(self, index, x0, y0, t0, level, mother_index, weight): 
        self.index = index
        self.array = np.zeros((3,1))
        self.array[0,0] = x0
        self.array[1,0] = y0
        self.array[2,0] = t0
        self.level = level
        self.mother_index = mother_index
        self.weight = weight
        
    def add_point(self,x,y,t):
        col = np.array([[x],[y],[t]])
        self.array=  np.append(self.array,col,axis=1)
        
    def del_point(self,i):
        self.array = np.delete(self.array,i,axis=1)
        
        
class VeinJunction:
    
    def __init__(self,time,array,level,mother_index):
        self.array=array
        self.time=time
        self.level=level
        self.mother_index=mother_index
        

### TOOLBOX

def initial_array_line(nb_points_0 = NB_POINTS_0, noise_reduction = NOISE_REDUCTION, x_axis = X_AXIS, nb_veins = NB_VEINES):
    
    first_array = np.zeros((10,nb_points_0))
    first_array[8,:] = 0
    index = 0
    
    for x_coordinate in x_axis:
        first_array[0,index] = (x_coordinate + np.random.normal() * noise_reduction)
        # écart entre les points = 1 de base
        # L = 100
        # nb de points = 100
        # noise reduction = 1/20, 1/50 1/100, 1/200 etc.
        # 
        index += 1
        
    first_array, list_veins, list_junctions = initiate_veins(first_array, nb_veins)
    
    first_array = distance_array(first_array)
    first_array = vecteur_normal_corrected(first_array)
    first_array = curvature_corrected(first_array)
    
    print(f"The first array is done")
    return(first_array, list_veins, list_junctions)


def initial_array_arc(noise_reduction = NOISE_REDUCTION, nb_points = NB_POINTS_0, R = X_AXIS[-1]):
    first_array = np.zeros((10,nb_points))
    theta = np.linspace(3 * np.pi / 2, 0, nb_points, )
    for i in range(0,len(theta)):
        theta[i] = theta[i] + np.random.normal() * noise_reduction
    first_array[0,:] = R * (1 + np.cos(theta))
    first_array[1,:] = R * np.sin(theta)
    
    first_array = distance_array(first_array)
    first_array = vecteur_normal_corrected(first_array)
    first_array = curvature_corrected(first_array)    
    
    return(first_array)

def initiate_veins(first_array, nb_veins, basal_weight = BASAL_WEIGHT, central_weight = CENTRAL_WEIGHT):
    
    # Initiate
    
    veins_index = []
    list_veins = []
    list_junctions = []
    
    # Determine and veins index and add in first array
    
    nb_points = first_array.shape[1]
    for vein in range(nb_veins):
        index = int(nb_points*(vein+1) / (nb_veins+1))
        veins_index.append(index)
        first_array[8,index] = vein  
    first_array[7,veins_index] = 1
    
    # Find central vein
    
    center = int(nb_points / 2)
    output_list = [None] * len(veins_index)
    output_list[:] = [abs(number - center) for number in veins_index]
    middle_index_listvein = output_list.index(min(output_list))  #use the index method to find the index in list_vein
    middle_index_inarray = veins_index[middle_index_listvein]  #what is the equivalent index in the array  
    
    # Initiate vein in class/list_veins
    
    weight_vec = [basal_weight] * len(veins_index)
    weight_vec[middle_index_listvein] = central_weight
    
    for vein in range(len(veins_index)):
        #vein(self, index, x0, y0, t0, level, mother_index, weight)
        v = VeinArray(vein,first_array[0,veins_index[vein]], first_array[1,veins_index[vein]], 0, 0, np.nan, weight_vec[vein])
        list_veins.append(v)  
        txt = "initial vein creation: vein index is {} and weight is {}"
        print(txt.format(v.index,v.weight))
        
    print(f"Veins were created")
    return(first_array,list_veins,list_junctions)

def distance_array(growth_front):
    
    growth_front[2,0] = 0
    for point in range(1,growth_front.shape[1]):
        growth_front[2,point] = math.sqrt((growth_front[0,point] - growth_front[0,point-1])**2 + (growth_front[1,point] - growth_front[1,point-1])**2) 
        
    return(growth_front)


def vecteur_normal_corrected(growth_front): 
    #divided the value by 2, except at the border
    
    #first point
    
    growth_front[3,0] = growth_front[1,1] - growth_front[1,0]  #u
    growth_front[4,0] = - growth_front[0,1] + growth_front[0,0]  #v
    Lo = math.sqrt(growth_front[3,0]**2 + growth_front[4,0]**2)  #normalized
    growth_front[3,0] = growth_front[3,0] / Lo
    growth_front[4,0] = growth_front[4,0] / Lo
    
    #front Loop
    
    for point in range(1,growth_front.shape[1] - 1):
            growth_front[3,point] = (growth_front[1,point+1] - growth_front[1,point-1]) / 2  #have to divide the distance by two
            growth_front[4,point] = (-growth_front[0,point+1] + growth_front[0,point-1]) / 2
            Lo = math.sqrt(growth_front[3,point]**2 + growth_front[4,point]**2)  #normalized
            growth_front[3,point] = growth_front[3,point] / Lo
            growth_front[4,point] = growth_front[4,point] / Lo   
            
    #Last point
    
    growth_front[3,-1] = growth_front[1,-1] - growth_front[1,-2]
    growth_front[4,-1] = -growth_front[0,-1] + growth_front[0,-2]
    Lo = math.sqrt(growth_front[3,-1]**2 + growth_front[4,-1]**2)  #normalized
    growth_front[3,-1] = growth_front[3,-1] / Lo
    growth_front[4,-1] = growth_front[4,-1] / Lo
    
    # Warning, for the direction (inside or outside)
    
    growth_front[4,:] = - growth_front[4,:]
    growth_front[3,:] = - growth_front[3,:]
    
    return(growth_front)

def curvature_corrected(growth_front): 
    
    # first point
    u_1 = growth_front[3,0]
    v_1 = growth_front[4,0]
    u_2 = growth_front[3,1]
    v_2 = growth_front[4,1]
    corr = 1 / 2
    d = growth_front[2,1] * corr
    growth_front[9,0] = math.asin(u_1 * v_2 - u_2 * v_1) / d
    
    # second point
    u_1 = growth_front[3,0]
    v_1 = growth_front[4,0]
    u_2 = growth_front[3,2]
    v_2 = growth_front[4,2]
    corr = 3 / 4
    d = (growth_front[2,1] + growth_front[2,2]) * corr
    growth_front[9,1] = math.asin(u_1 * v_2 - u_2 * v_1) / d    
    
    
    # loop     
    for point in range(2,growth_front.shape[1] - 2):
        u_1 = growth_front[3,point-1]
        v_1 = growth_front[4,point-1]
        u_2 = growth_front[3,point+1]
        v_2 = growth_front[4,point+1]
        d = growth_front[2,point+1] + growth_front[2,point]             
        growth_front[9,point] = math.asin(u_1 * v_2 - u_2 * v_1) / d
    
    # first before last
    u_1 = growth_front[3,-3]
    v_1 = growth_front[4,-3]
    u_2 = growth_front[3,-1]
    v_2 = growth_front[4,-1]
    corr = 3 / 4
    d = (growth_front[2,-1] + growth_front[2,-2]) * corr
    growth_front[9,-2] = math.asin(u_1 * v_2 - u_2 * v_1) / d  
    
    # last point
    v_1=growth_front[3,-1]
    u_1=growth_front[4,-1]
    v_2=growth_front[3,-2]
    u_2=growth_front[4,-2]
    corr = 1 / 2
    d = growth_front[2,-1] * corr
    growth_front[9,-1] = math.asin(u_1 * v_2 - u_2 * v_1) / d 
    
    return(growth_front)


def growth_frond_const(growth_front):
    
    gf_deepcopy = deepcopy(growth_front) 
    A_L=0.2
    gf_deepcopy[0,:] = gf_deepcopy[0,:] + A_L * growth_front[3,:]
    gf_deepcopy[1,:] = gf_deepcopy[1,:] + A_L * growth_front[4,:]
    
    gf_deepcopy = distance_array(gf_deepcopy)
    gf_deepcopy = vecteur_normal_corrected(gf_deepcopy)
    gf_deepcopy = curvature_corrected(gf_deepcopy)   
    
    return(gf_deepcopy)


def growth_vein_dependant(growth_front, a0 = A0, am = AM, s_growth = S_GROWTH , d_vein = D_VEIN, dt = DT):
    
    #Copy initial array
    next_growth_front = deepcopy(growth_front)  

    #Dependant Growth
    veins = np.nonzero(growth_front[7,:] == 1)[0]
    s = np.cumsum(growth_front[2,:])

    delta_var = np.zeros(next_growth_front.shape[1]) + a0
    for j in veins:   
        dj = s - s[j]
        A_j = am * np.exp(-(dj**2) / (s_growth**2))
        delta_var += A_j
        #plt.plot(delta_var)  #check if error in growth
    delta_var = delta_var * (s_growth / d_vein)  # Normalized

    # Compute x and y
    next_growth_front[0,:] = growth_front[0,:] + (delta_var * dt) * growth_front[3,:]
    next_growth_front[1,:] = growth_front[1,:] + (delta_var * dt) * growth_front[4,:]
    
    next_growth_front = distance_array(next_growth_front)
    next_growth_front = vecteur_normal_corrected(next_growth_front)
    next_growth_front = curvature_corrected(next_growth_front)  
  
    return(next_growth_front)


@jit
def angle_pol(x_old,y_old,x_new,y_new,ind):
    
    # INSIDE POINT COORDINATES
    x_ins = x_new[ind]
    y_ins = y_new[ind]

    ## SUM OF ANGLES

    # loop angles
    sum_angle=0
    for i in range(len(x_old)):
        if i == len(x_old)-1:
            vector1_x  = x_old[i] - x_ins
            vector1_y = y_old[i] - y_ins

            vector2_x = x_old[0] - x_ins
            vector2_y = y_old[0] - y_ins

            angle = math.atan2( vector1_x*vector2_y - vector1_y*vector2_x, vector1_x*vector2_x + vector1_y*vector2_y)
            sum_angle += angle
        else: 
            vector1_x = x_old[i] - x_ins
            vector1_y = y_old[i] - y_ins

            vector2_x = x_old[i+1] - x_ins
            vector2_y = y_old[i+1] - y_ins

            angle = math.atan2( vector1_x*vector2_y - vector1_y*vector2_x, vector1_x*vector2_x + vector1_y*vector2_y)
            sum_angle += angle
 
    return(sum_angle/math.pi)


def growth_vein_obstacle(growth_front, a0 = A0, am = AM, s_growth = S_GROWTH , d_vein = D_VEIN, dt = DT):
    
    #Copy initial array
    old_front = deepcopy(growth_front)
    next_growth_front = deepcopy(growth_front)  

    #Dependant Growth
    veins = np.nonzero(growth_front[7,:] == 1)[0]
    s = np.cumsum(growth_front[2,:])

    delta_var = np.zeros(next_growth_front.shape[1]) + a0
    for j in veins:   
        dj = s - s[j]
        A_j = am * np.exp(-(dj**2) / (s_growth**2))
        delta_var += A_j
        #plt.plot(delta_var)  #check if error in growth
    delta_var = delta_var * (s_growth / d_vein)  # Normalized

    # Compute x and y
    next_growth_front[0,:] = growth_front[0,:] + (delta_var * dt) * growth_front[3,:]
    next_growth_front[1,:] = growth_front[1,:] + (delta_var * dt) * growth_front[4,:]
    
    #  find overlap and restore
    
    ind_toremove = []
    for i in range(1, old_front.shape[1] - 1):
        angle_i = angle_pol(old_front[0,:], old_front[1,:], next_growth_front[0,:], next_growth_front[1,:], i)
        if (angle_i > 0.1) or (angle_i < -0.1):
            ind_toremove.append(i)
            
            
    next_growth_front = np.delete(next_growth_front, ind_toremove, axis = 1)
    
    
    next_growth_front = distance_array(next_growth_front)
    next_growth_front = vecteur_normal_corrected(next_growth_front)
    next_growth_front = curvature_corrected(next_growth_front)  
  
    return(next_growth_front)

@jit
def growth_vein_obweight(growth_front, list_veins, a0 = A0, am = AM, s_growth = S_GROWTH , d_vein = D_VEIN, dt = DT, threshold = 3):
    
    #Copy initial array
    old_front = deepcopy(growth_front)
    next_growth_front = deepcopy(growth_front)  

    #Dependant Growth
    veins = np.nonzero(growth_front[7,:] == 1)[0]
    s = np.cumsum(growth_front[2,:])

    delta_var = np.zeros(next_growth_front.shape[1]) + a0
    for j in veins:   
        dj = s - s[j]
        vein_index = int(old_front[8,j])
        A_j = am * list_veins[vein_index].weight * np.exp(-(dj**2) / (s_growth**2))  #modified to take weight into account
        delta_var += A_j
        #plt.plot(delta_var)  #check if error in growth
    delta_var = delta_var * (s_growth / d_vein)  # Normalized

    # Compute x and y
    next_growth_front[0,:] = growth_front[0,:] + (delta_var * dt) * growth_front[3,:]
    next_growth_front[1,:] = growth_front[1,:] + (delta_var * dt) * growth_front[4,:]
    
    #  find overlap and restore
    
    ind_toremove = []
    for i in range(1, old_front.shape[1] - 1):
        angle_i = angle_pol(old_front[0,:], old_front[1,:], next_growth_front[0,:], next_growth_front[1,:], i)
        if (angle_i > 0.1) or (angle_i < -0.1):
            ind_toremove.append(i)
            
    consecutive = []
    if len(ind_toremove) >= threshold:
        for i, ind in enumerate(ind_toremove):

            # as long as a list starting from ind would not be out of bound
            if i <= len(ind_toremove) - threshold:

                # creation of list to compare 
                ls_threshold = []
                for j in range(threshold):
                    ls_threshold.append(ind_toremove[i + j])
                    
                # creation of an sequence to compare
                ls_consecutive = []
                for j in range(threshold):
                    ls_consecutive.append(ind_toremove[i] + j)

                # are the sequence equal 
                if ls_threshold == ls_consecutive:
                    next_growth_front[5, ls_consecutive] = 2
                    consecutive.append(ls_consecutive)

    if len(consecutive) >= 1:
        consecutive = np.array(consecutive)
        consecutive = np.unique(consecutive)
        consecutive = list(consecutive)
        
    # to not delete border points
    ind_toremove = [i for i in ind_toremove if i not in consecutive] 
    
    
    # to put border point growth to zero 
    #next_growth_front[:1, consecutive] = old_front[:1, consecutive]
    
    # delete loops
    next_growth_front = np.delete(next_growth_front, ind_toremove, axis = 1)
    
    # compute classic parameters
    next_growth_front = distance_array(next_growth_front)
    next_growth_front = vecteur_normal_corrected(next_growth_front)
    next_growth_front = curvature_corrected(next_growth_front)  

    return(next_growth_front)


def theta(growth_front):
    
    theta_front = []
    u, v = list(growth_front[3,:]), list(growth_front[4,:])
    
    for i in range(growth_front.shape[1]):
        if v[i] >= 0:
            theta_front.append(math.atan(u[i] / v[i]))
            
        else:
            if u[i] >= 0:
                theta_front.append(math.atan(u[i] / v[i]) + math.pi )
            else:
                theta_front.append(math.atan(u[i] / v[i]) - math.pi )
    
    return(theta_front)


def f_theta_exp(x, th_0 = 80, th_1 = 100, measure = 'radians'):
    
    
    if measure == 'radians':
        th_0 = np.deg2rad(th_0)
        th_1 = np.deg2rad(th_1)

    if isinstance(x,np.ndarray) or isinstance(x, list):
        if isinstance(x, list):
            x = np.array(x)
       
        ins = np.where(np.abs(x) >= th_0)[0]
        out = np.where(np.abs(x) < th_0)
        y = x.copy()
        y[ins] = np.exp(- (np.abs(x[ins])- np.abs(th_0))**2 / (np.abs(th_1) - np.abs(th_0))**2)
        y[out] = 1
        
        
    elif isinstance(x,int):
        if abs(x) >= th_0:
            y = np.exp(- (abs(x)- abs(th_0))**2 / (abs(th_1) - abs(th_0))**2)
        else:
            y = 1
            
    else:
        return print("ERROR: not an array list or int")
    
    return  y


def f_theta_sig(x, th_0 = 90, delta_th = 40, measure = 'radians'):
    
    if measure == 'radians':
        th_0 = np.deg2rad(th_0)
        delta_th = np.deg2rad(delta_th)

    if isinstance(x, list):
        x = np.array(x)    
    
    u = (np.abs(x) - th_0) / delta_th
    y = 1 / (1 + np.exp(u))
        
    return(y)
    

@jit
def growth_vein_angle(growth_front, list_veins, iteration, a0 = A0, am = AM, s_growth = S_GROWTH , d_vein = D_VEIN, dt = DT, threshold = 3):
    
    # Copy initial array
    old_front = deepcopy(growth_front)
    next_growth_front = deepcopy(growth_front)  

    # Compute theta and f(theta)

    theta_front = theta(old_front)
    f_theta = f_theta_sig(theta_front)
    
    
    # Dependant Growth
    veins = np.nonzero(growth_front[7,:] == 1)[0]
    s = np.cumsum(growth_front[2,:])

    delta_var = np.zeros(next_growth_front.shape[1]) + a0
    for j in veins:   
        dj = s - s[j]
        vein_index = int(old_front[8,j])
        A_j = am * list_veins[vein_index].weight * np.exp(-(dj**2) / (s_growth**2))  #modified to take weight into account
        delta_var += A_j
        #plt.plot(delta_var)  #check if error in growth
    delta_var = f_theta * delta_var * (s_growth / d_vein)  # Normalized
    

    # Compute x and y
    next_growth_front[0,:] = growth_front[0,:] + (delta_var * dt) * growth_front[3,:]
    next_growth_front[1,:] = growth_front[1,:] + (delta_var * dt) * growth_front[4,:]
    
    # find overlap and restore
    
    ind_toremove = []
    for i in range(1, old_front.shape[1] - 1):
        angle_i = angle_pol(old_front[0,:], old_front[1,:], next_growth_front[0,:], next_growth_front[1,:], i)
        if (angle_i > 0.1) or (angle_i < -0.1):
            ind_toremove.append(i)
            
    consecutive = []
    if len(ind_toremove) >= threshold:
        for i, ind in enumerate(ind_toremove):

            # as long as a list starting from ind would not be out of bound
            if i <= len(ind_toremove) - threshold:

                # creation of list to compare 
                ls_threshold = []
                for j in range(threshold):
                    ls_threshold.append(ind_toremove[i + j])
                    
                # creation of an sequence to compare
                ls_consecutive = []
                for j in range(threshold):
                    ls_consecutive.append(ind_toremove[i] + j)

                # are the sequence equal 
                if ls_threshold == ls_consecutive:
                    next_growth_front[5, ls_consecutive] = 2
                    consecutive.append(ls_consecutive)

    if len(consecutive) >= 1:
        consecutive = np.array(consecutive)
        consecutive = np.unique(consecutive)
        consecutive = list(consecutive)
        
    # to not delete border points
    ind_toremove = [i for i in ind_toremove if i not in consecutive] 
    
    
    # to put border point growth to zero 
    #next_growth_front[:1, consecutive] = old_front[:1, consecutive]
    
    # delete loops
    next_growth_front = np.delete(next_growth_front, ind_toremove, axis = 1)
    
    # compute classic parameters
    next_growth_front = distance_array(next_growth_front)
    next_growth_front = vecteur_normal_corrected(next_growth_front)
    next_growth_front = curvature_corrected(next_growth_front)  

    return(next_growth_front)


def insert_points(growth_front, max_res = MAX_RES):
    
    if growth_front.shape[1] < 10000:  #maximum size
        insert = np.nonzero(growth_front[2,:] > max_res)[0]
        #print('insert len' , len(insert))
        #print('insert', insert)
        growth_front[5,:] = 0
        if len(insert) != 0:
            col = np.zeros((growth_front.shape[0], len(insert)))
            for j in range(0, len(insert)):
                col[0,j] = (growth_front[0,insert[j]] + growth_front[0,insert[j]-1]) / 2
                col[1,j] = (growth_front[1,insert[j]] + growth_front[1,insert[j]-1]) / 2
                col[5,j] = 1
            growth_front = np.insert(growth_front, insert[:], col, axis = 1)
            
    growth_front = distance_array(growth_front)
    growth_front = vecteur_normal_corrected(growth_front)
    growth_front = curvature_corrected(growth_front) 
 
    return(growth_front)

def delete_points(growth_front, min_res = MIN_RES):
    
    delete = np.nonzero((growth_front[2,:] < min_res) & (growth_front[7,:] == 0))[0]
    
    while len(delete) > 1:  #reminder: after a test, you showed you were deleting too many points if using directly np.delete
        growth_front = np.delete(growth_front,delete[1], axis = 1)
        growth_front = distance_array(growth_front)
        delete = np.nonzero((growth_front[2,:] < min_res) & (growth_front[7,:] == 0))[0] #delete point by point
        
    growth_front = distance_array(growth_front)
    growth_front = vecteur_normal_corrected(growth_front)
    growth_front = curvature_corrected(growth_front) 
    
    return(growth_front)


def loop_removal(old_front, new_front, iteration):
    
    #  create matrix with x and y coordinates
    old_mat = np.array([old_front[0,:], old_front[1,:]])
    new_mat = np.array([new_front[0,:], new_front[1,:]])
    
    #  compute the list of distance bewteen a point and itself at t + 1, find max spread to consider
    spread_list = np.sqrt(np.sum((new_mat - old_mat)**2, axis = 0))
    max_spread = max(spread_list)
    d_points = min(new_front[2,1:])  #the distance at the first point is zero
    #neighbours = math.ceil(2 * max_spread / d_points)
    neighbours = old_front.shape[1]
    
    #  create and fill distance matrice, lines: (size 2*neighbours + 1), columns: (number of points columns)
    #  each lines correspond to the distance between all the points of the front and the neighbour k
    dist_mat = np.zeros((2 * neighbours + 1, np.shape(new_mat)[1]))
    for k in range(-neighbours, neighbours + 1):
        old_mat_rolled = np.roll(old_mat, k)
        dist_mat[neighbours + k] = np.sqrt(np.sum((new_mat - old_mat_rolled)**2, axis = 0))
        
    if iteration % 100 == 0:
        print('dist mat dimensions', dist_mat.ndim)
        print('min dist_mat:', dist_mat.min())
        print('max dist_mat:', dist_mat.max())
        print('mean dist_mat', dist_mat.mean())
        print('std dist_mat', np.std(dist_mat))
        print('spread_list dimension', spread_list.ndim)
        print('min spread_list', spread_list.min())
        print('max spread_list', spread_list.max())
        print('mean spread list', spread_list.mean())
        print('std spread list', np.std(spread_list))
    
    #  delete a point of the new front if there is 
    #  at least one point for which the distance is inferior to the spread of the point
    
    ind_toremove = np.unique((dist_mat < spread_list[None,:]).nonzero()[1])
    
    #  now delete it
    cleaned_front = np.delete(new_front, ind_toremove, axis = 1)
    
    #  compute other parameters to give clean array
    cleaned_front = distance_array(cleaned_front)
    cleaned_front = vecteur_normal_corrected(cleaned_front)
    cleaned_front = curvature_corrected(cleaned_front) 
    
    return(cleaned_front)


def propagate_veins_regular(growth_front, list_veins, list_junctions, iteration_time, d_vein = D_VEIN):
 
    ##  DELETE PREVIOUS JUNCTIONS

    erease = np.nonzero(growth_front[7,:] > 1)[0]
    growth_front[7,erease] = 0

    ##  COMPUTE DISTANCES BETWEEN VEINS
    
    distance = 0
    for point in range(growth_front.shape[1]): 
        
        #  no vein
        
            if growth_front[7,point] == 0:
                distance = distance + growth_front[2,point]
                growth_front[6,point] = distance 
        
        #  vein          
        
            elif growth_front[7,point] == 1:
                distance = distance + growth_front[2,point]
                growth_front[6,point] = distance
                distance = 0
                
    ##  VEIN CREATION
    
    veins = np.nonzero(growth_front[7,:] == 1)[0]  #  find all veins
    
    #  if a single vein, we take into account the distance to the edge
    
    if len(veins) <= 1:
        v0 = 0
        v1 = veins[0]
        v2 = -1

        #  critical distance rule
         
        if growth_front[6,v1] + growth_front[6,v2] >= 2 * d_vein:
            growth_front[7,v1] = 0
            third_1 = int(v0 + 2*(v1 - v0) / 3)
            third_2 = int(v1 + (v2 - v1) / 3)    
            growth_front[7,third_1] = 1
            growth_front[7,third_2] = 1
            growth_front[8,third_1] = len(list_veins)
            growth_front[8,third_2] = len(list_veins) + 1
            growth_front[7,third_1+1:third_2-1] = 2
            
            #  new vein in list_veins     
            
            mother_index = int(growth_front[8,v1])
            growth_front[8,v1] = 0
            level = list_veins[mother_index].level+1
            v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index)
            v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index)
            list_veins.append(v_third_1)
            list_veins.append(v_third_2)           
             
            #  junctions creation
            
            array_junction = growth_front[0:2,third_1+1:third_2-1]
            link = VeinJunction(iteration_time, array_junction, level, mother_index)
            list_junctions.append(link)
            third_1 = None
            third_2 = None
            
    #  if multiple veins
        
    else:
            for vein_index in range(0,len(veins)):
                
                if vein_index == 0:  
                    v0 = 0
                    v1 = veins[vein_index]
                    v2 = veins[vein_index+1]
                    
                elif vein_index == len(veins) - 1:
                    v0 = veins[vein_index-1]
                    v1 = veins[vein_index]
                    v2 = growth_front.shape[1] - 1
                    
                else: 
                    v0 = veins[vein_index-1]
                    v1 = veins[vein_index]
                    v2 = veins[vein_index+1]
            
                #  critical distance rule
                
                if growth_front[6,v1] + growth_front[6,v2] >= 2 * d_vein:
                    growth_front[7,v1] = 0
                    third_1 = int(v0+2*(v1-v0)/3)
                    third_2 = int(v1+(v2-v1)/3)
                    growth_front[7,third_1] = 1
                    growth_front[7,third_2] = 1
                    growth_front[8,third_1] = len(list_veins)
                    growth_front[8,third_2] = len(list_veins)+1
                    growth_front[7,third_1+1:third_2-1] = 2
            
                #  new vein in list_veins      
                
                    mother_index = int(growth_front[8,v1])
                    growth_front[8,v1] = 0
                    level = list_veins[mother_index].level + 1
                    v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index)
                    v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index)
                    list_veins.append(v_third_1)
                    list_veins.append(v_third_2)
            
                #  junctions creation
            
                    array_junction = growth_front[0:2,third_1+1:third_2-1]
                    link = VeinJunction(iteration_time, array_junction, level, mother_index)
                    list_junctions.append(link)
                    third_1 = None
                    third_2 = None
        
 ## PROPAGATE VEINS
    where_veins = np.where(growth_front[7,:] == 1)[0]
    
    for v in where_veins:
        list_veins[int(growth_front[8,v])].add_point(growth_front[0,v], growth_front[1,v], iteration_time)
     
    
    return(growth_front, list_veins, list_junctions)


def propagate_veins_weight(growth_front, list_veins, list_junctions, iteration_time, d_vein = D_VEIN, central_weight = CENTRAL_WEIGHT, basal_weight = BASAL_WEIGHT):
 
    ##  DELETE PREVIOUS JUNCTIONS

    erease = np.nonzero(growth_front[7,:] > 1)[0]
    growth_front[7,erease] = 0
    
    #mother_index = growth_front[8,:].max()
    #if mother_index > len(list_veins):
        #print('av prop mother_index max:', mother_index)
        #print('av prop len list veins:', len(list_veins))

    ##  COMPUTE DISTANCES BETWEEN VEINS
    
    distance = 0
    for point in range(growth_front.shape[1]): 
        
        #  no vein
        
            if growth_front[7,point] == 0:
                distance = distance + growth_front[2,point]
                growth_front[6,point] = distance 
        
        #  vein          
        
            elif growth_front[7,point] == 1:
                distance = distance + growth_front[2,point]
                growth_front[6,point] = distance
                distance = 0
                
    ##  VEIN CREATION
    
    veins = np.nonzero(growth_front[7,:] == 1)[0]  #  find all veins
    
    #  if a single vein, we take into account the distance to the edge
    
    if len(veins) <= 1:
        v0 = 0
        v1 = veins[0]
        v2 = -1

        #  critical distance rule
         
        if growth_front[6,v1] + growth_front[6,v2] >= 2 * d_vein:
            growth_front[7,v1] = 0
            third_1 = int(v0 + 2*(v1 - v0) / 3)
            third_2 = int(v1 + (v2 - v1) / 3)    
            growth_front[7,third_1] = 1
            growth_front[7,third_2] = 1
            growth_front[8,third_1] = len(list_veins)
            growth_front[8,third_2] = len(list_veins) + 1
            growth_front[7,third_1 + 1:third_2-1] = 2
            
            #  new vein in list_veins     
            
            mother_index = int(growth_front[8,v1])
            growth_front[8,v1] = 0
            level = list_veins[mother_index].level+1
            mother_weight = list_veins[mother_index].weight 
            
            if mother_weight == basal_weight:
                v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, mother_weight)
                v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, mother_weight)
        
            elif mother_weight == central_weight:
                
                #  find which vein is closer to the center (on x)
                
                absx_v1 = abs(growth_front[0,third_1])
                absx_v2 = abs(growth_front[0,third_2])
                              
                if absx_v1 < absx_v2 :
                    v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, central_weight)
                    v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, basal_weight)
                elif absx_v1 > absx_v2 :
                    v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, basal_weight)
                    v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, central_weight)
                else:
                    rnumb = random.randint(0,1)  #in case the distance to the center is equal
                    if rnumb == 0:
                        v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, basal_weight)
                        v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, central_weight)
                    elif rnumb == 1:
                        v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, central_weight)
                        v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, basal_weight)
                   
            list_veins.append(v_third_1)
            list_veins.append(v_third_2)           
             
            #  junctions creation
            
            array_junction = growth_front[0:2,third_1+1:third_2-1]
            link = VeinJunction(iteration_time, array_junction, level, mother_index)
            list_junctions.append(link)
            third_1 = None
            third_2 = None
            
    #  if multiple veins
        
    else:
            for vein_index in range(0,len(veins)):
                
                if vein_index == 0:  
                    v0 = 0
                    v1 = veins[vein_index]
                    v2 = veins[vein_index + 1]
                    
                elif vein_index == len(veins) - 1:
                    v0 = veins[vein_index-1]
                    v1 = veins[vein_index]
                    v2 = growth_front.shape[1] - 1
                    
                else: 
                    v0 = veins[vein_index-1]
                    v1 = veins[vein_index]
                    v2 = veins[vein_index+1]
            
                #  critical distance rule
                
                mother_index = int(growth_front[8,v1])
                if growth_front[6,v1] + growth_front[6,v2] >= 2 * d_vein:
                    growth_front[7,v1] = 0
                    third_1 = int(v0+2*(v1-v0)/3)
                    third_2 = int(v1+(v2-v1)/3)
                    growth_front[7,third_1] = 1
                    growth_front[7,third_2] = 1
                    growth_front[8,third_1] = len(list_veins)
                    growth_front[8,third_2] = len(list_veins)+1
                    growth_front[7,third_1+1:third_2-1] = 2
            
                #  new vein in list_veins      

                    if mother_index > len(list_veins):
                        print('growth_front[8,v1]', growth_front[8,v1])
                        print('v1', v1)
                        print('v0', v0)
                        print('v2', v2)
                        print('it time', iteration_time)
                        print('third 1', third_1)
                        print('third 2', third_2)
                        print('growth_front[6,v1]', growth_front[6,v1])
                        print('growth_front[6,v2]', growth_front[6,v2])
                        print('d_vein', d_vein)

                        print('mother_index ap prop', mother_index)
                        print('len list veins ap prop', len(list_veins))
                        

                        plt.scatter(growth_front[0,v0:v2], growth_front[1,v0:v2])
                        plt.scatter(growth_front[0,v1], growth_front[1,v1], color = 'red')
                        plt.scatter(growth_front[0,v0], growth_front[1,v0], color = 'red')
                        plt.scatter(growth_front[0,v2], growth_front[1,v2], color = 'red')
                        
                        
                        #mother = [vein.mother_index for vein in list_veins]
                        #print('mother:', mother)
                    
                    growth_front[8,v1] = 0
                    level = list_veins[mother_index].level+1
                    mother_weight = list_veins[mother_index].weight 
            
                    if mother_weight == basal_weight:
                        v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, mother_weight)
                        v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, mother_weight)

                    elif mother_weight == central_weight:

                        #  find which vein is closer to the center (on x)

                        absx_v1 = abs(growth_front[0,third_1])
                        absx_v2 = abs(growth_front[0,third_2])

                        if absx_v1 < absx_v2 :
                            v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, central_weight)
                            v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, basal_weight)
                        elif absx_v1 > absx_v2 :
                            v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, basal_weight)
                            v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, central_weight)
                        else:
                            rnumb = random.randint(0,1)  #in case the distance to the center is equal
                            if rnumb == 0:
                                v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, basal_weight)
                                v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, central_weight)
                            elif rnumb == 1:
                                v_third_1 = VeinArray(len(list_veins), growth_front[0,third_1], growth_front[1,third_1], iteration_time, level, mother_index, central_weight)
                                v_third_2 = VeinArray(len(list_veins) + 1, growth_front[0,third_2], growth_front[1,third_2], iteration_time, level, mother_index, basal_weight)

                    list_veins.append(v_third_1)
                    list_veins.append(v_third_2)
            
                #  junctions creation
            
                    array_junction = growth_front[0:2,third_1+1:third_2-1]
                    link = VeinJunction(iteration_time, array_junction, level, mother_index)
                    list_junctions.append(link)
                    third_1 = None
                    third_2 = None
        
 ## PROPAGATE VEINS
    where_veins = np.where(growth_front[7,:] == 1)[0]
    
    for v in where_veins:
        list_veins[int(growth_front[8,v])].add_point(growth_front[0,v], growth_front[1,v], iteration_time)
     
    
    return(growth_front, list_veins, list_junctions)


### SAVE AND READ

def write_arraylist(arraylist, filename):

    #  create empty name of zero
    zeros = '0' * len(str(len(arraylist)))
    
    #  open fil to write
    f = tables.open_file(filename, 'w')
                       
    #  add array with the right name
    for i, array in enumerate(arraylist):
        
        #create name
        name = list(zeros) 
        item = list(str(i))
        
        for j, letter in enumerate(item[::-1]):
            index = - (j + 1)
            name[index] = letter
        name = "".join(name)
        name = "a" + name
        
        # add array to the file
        f.create_array(f.root, name, array)
        
    #  close file
    f.close()
    
    
def read_array_list(filename):

    list_array = []
    with h5py.File(filename, "r") as f:
        #  Print all root level object names (aka keys) 
        #  these can be group or dataset names 
        print("Keys: %s" % f.keys())
        for key_index in range(len(f.keys())):
        #  get 'key' object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[key_index]            
        #  get the object type for a_group_key: usually group or dataset
            print(type(f[a_group_key]))         
        #  if a_group_key is a group name, 
        #  this gets the object names in the group and returns as a list
            data = list(f[a_group_key])
        #  If a_group_key is a dataset name, 
        #  this gets the dataset values and returns as a list    
            data = list(f[a_group_key])
        #  preferred methods to get dataset values:
            ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array
        
        # save array in list_array
            list_array.append(ds_arr)
            
    return(list_array)


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                return pickle.load(f)
            except EOFError:
                break
