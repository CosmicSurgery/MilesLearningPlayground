'''


'''

import numpy as np
import pandas as pd

data = {}

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
'''
CELL TYPE DATA

keys:
    >rs, fs, lts, ib - cell types
        >abdvui - returns the a b c d v u I parameters for a generic neuron of that cell_type
        >Prob_dist_r - returns an array specifying the distribution of a neuron's connection probability as a function of distance 
        >Prob_dist_type - returns a dictionary of connection probabilites as a function of the target neuron type
        >size - radius
color - color
'''
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
rs = {}
fs = {}
lts = {}
ib = {}

'''
Izhikevich Parameters for each cell type
'''
key = 'abcdvui'

# Regular Spiking - Pyramidal
a = 0.02
b = 0.2
c = -65
d = 8
v = -65
u = v*b
I = [0]
rs[key] = (a,b,c,d,v,u,I)

# Fast Spiking - Parvalbumin
a = 0.02
b = 0.25
c = -65
d = 2
v = -65
u = v*b
I = [0]
fs[key] = (a,b,c,d,v,u,I)

# Low Threshold Spiking - Somatostatin
a = 0.01
b = 0.2
c = -65
d = 2
v = -65
u = v*b
I = [0]
lts[key] = (a,b,c,d,v,u,I)

# Intrinsic Burst - VIP
a = 0.02
b = 0.2
c = -55
d = 4
v = -65
u = v*b
I = [0]
ib[key] = (a,b,c,d,v,u,I)

#Probability of a synapse forming as a function of distance from 0 to 1 units

key = 'Prob_dist_r'

small = 12
medium = 10
large = 8
x= np.arange(0,1,0.001)

rs[key] = [np.e**(-small*i) for i in x]
fs[key] = [np.e**(-large*i) for i in x]
lts[key] = [np.e**(-large*i) for i in x]
ib[key] = [np.e**(-large*i) for i in x]

#Probability of a synapse forming as a function of cell_type

key = 'Prob_dist_type'

rs[key] = { 'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.8, 'ib' : 0.8}
fs[key] = {'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.0, 'ib' : 0.0}
lts[key] = {'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.0, 'ib' : 0.0}
ib[key] = {'rs' : 0.0, 'fs' : 0.3, 'lts' : 0.8, 'ib' : 0.0}

# Cell size
key = 'size'

rs[key] = 50 / 1000
fs[key] = 50 / 1000
lts[key] = 50 / 1000
ib[key] = 50 / 1000

# Neuron type color
key = 'color'

rs[key]= 'dodgerblue'
fs[key]= 'firebrick'
lts[key]= 'forestgreen'
ib[key]= 'yellow'

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
'''
NETWORK LEVEL PARAMETERS
keys:
    >network
        >timing - T1, T2, step, sampling rate 
'''
key = 'network'
T1 = 0
T2 = 100 # milliseconds
h = 0.1
sf = 10 # kHz

data[key] = {(T1, T2, h, sf)}

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


info_keys = ['model type', 'connects to', 'conduction delay', 'connection strength', 'prolific']

rs_info_dict = {'model type' : 'pyramidal','connects to' : {'rs' : 0.25,'fs' : 0.25,'lts' : 0.25,'ib' : 0.25}, 'conduction delay' : 3.5, 'connection strength' : None, 'prolific' : 0.8, 'range' : None}

fs_info_dict = {'model type' : 'parvalbumin','connects to' : {'rs' : 0.5,'fs' : 0.5}, 'conduction delay' : 2, 'connection strength' : None, 'prolific' : 0.1, 'range' : None}

lts_info_dict = {'model type' : 'somatostatin','connects to' : {'rs' : 0.5,'fs' : 0.5}, 'conduction delay' : 3.5, 'connection strength' : None, 'prolific' : 0.06, 'range' : None}

ib_info_dict = {'model type' : 'vip','connects to' : {'fs' : 0.25,'lts' : 0.75}, 'conduction delay' : 3.5, 'connection strength' : None, 'prolific' : 0.04, 'range' : None}

neuron_info_dict = {'rs' : rs_info_dict, 'fs' : fs_info_dict, 'lts' : lts_info_dict, 'ib' : ib_info_dict}

# index 0 - rs, 1 - fs, 2 - lts, 3 - ib
connectivity = {'rs' : { 'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.8, 'ib' : 0.8}, 'fs' : {'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.0, 'ib' : 00}, 'lts' : {
    'rs' : 0.8, 'fs' : 0.8, 'lts' : 0.8, 'ib' : 0.8}, 'ib' : {'rs' : 0.0, 'fs' : 0.3, 'lts' : 0.8, 'ib' : 0.0}}

x = np.arange(0,1,0.001)

# Probability of a synapse forming at every distance x where step size is 1 micron (from 0 to 1000 microns)
cellRange = {'rs' : [np.e**(-10*i) for i in x], 'fs' : [np.e**(-10*i) for i in x], 'lts' : [np.e**(-10*i) for i in x], 'ib' : [np.e**(-10*i) for i in x]}

parameters = {'rs' : (0.02,0.2,-65,8), 'fs' : (0.02,0.25,-65,2), 'lts' : (0.1,0.2,-65,2), 'ib' : (0.02,0.2,-55,4)}

PSP_parameters = {'rs' : [31/32, 7/8, 7, 7], 'fs' : [31/32, 7/8, 5, 5], 'lts' : [31/32, 7/8, 5, 5], 'ib' : [31/32, 7/8, 5, 5]}

neuron_radius = 50 / 1000
noise= 0.25
sf = 10         # measured in khz

tuning_type = 'square root'


data= {'rs':rs, 'fs':fs, 'lts':lts,'ib':ib}