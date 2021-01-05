

import numpy as np
import random
'''
package from SNNconnectivity that has info on cell_type paramters and connectivity 
'''
from SNNconnectivity import neuron_info_dict, connectivity, cellRange, parameters, PSP_parameters, tuning_type

'''
PSP scaling and sizing info, not sure if these should stay global or if they can be moved inside the Neuron object
'''



neuron_radius = 50/1000

psp_slow_decay, psp_fast_decay = 31/32, 7/8
scaling_factor = np.floor((-np.log(np.log(psp_fast_decay)/np.log(psp_slow_decay)))/(np.log(psp_fast_decay)-np.log(psp_slow_decay)))
scaling_factor = (psp_slow_decay**scaling_factor - psp_fast_decay**scaling_factor)**(-1)
psp_potential = 10
psp_potential = -0.5 * np.ceil(psp_potential*scaling_factor)
scaling_factor=0.502         #y-value at psp-max
psp_potential = 10
psp_potential /= scaling_factor

'''
Neuron Object
'''

class Neuron:
    '''
    accepts neuron_radius and an index as option parameters probably redundant
    '''
    
    def __init__(self, neuron_radius = 50/1000, index = None, tuning_type = 'linear'):
        '''
        Neuron Object attributes that will automatically get applied to ever Neuron and any child from the Neuron class
        '''
        self.cell_type = None                                                               #stores a string such as 'rs', 'fs', 'lts', 'ib' to indicate cell type
        self.position = ((None, None))                                                      # stores the x,y position of each neuron 
        self.radius = neuron_radius                                                         #stores the radius of the neuron
        self.identity = None                                                                # Variable I used for debugging, I don't think it has a purpose anymore...
        self.index = index                                                                  # BUGGED - supposed to store the index to the neurons list (problem is there is such list for each cortex and each network)
        self.x,self.y = None, None                                                          # stores x and y positions respectively (not different from the 'position' attribute
        self.vMemory, self.uMemory, self.iMemory = [], [], []                               # Stores the memory of v, u and I (input) in lists from the ixhikevich modelcrete_Model, should be the same length as the total time of sim
        self.a,self.b,self.c,self.d = None, None, None, None                                # initializes the abcd paramters
        self.v, self.u, self.I = None, None, [0]                                                # initializes u and v as integer variables    
        self.h = 0.1                                                                        # step size in milliseconds
        self.fire = False                                                                   # boolean attribute that is True if the Neuron is currently firing, and False if it is not
        self.spikes = []                                                                    # spikes list stores the time whenever a spike occurs
        self.psp_slow_decay, self.psp_fast_decay, self.psp_slow_potential, self.psp_fast_potential  = None, None ,None, None      # stores the time constants for the psp waveform 
        self.static_input = False                                                           # Unused variable, intended for debuggin purposes
        self.is_an_input,self.is_an_output = 0, 0                                           # integers that store the amount of other neurons this neuron object connects to as either an input or output, used primary to eliminate islands and peninsulas in the neural network
        self.debug = 0                                                                      # debug attribute
        self.trimOK = True                                                                  # another debug attribute
        self.cortex = None                                                                  # defines which cortex this neuron belongs to
        self.rest_I = [0]
        self.tuning_type = tuning_type
        self.tuning_factor, self.tuning_direction = 0, 1 if random.random() < 0.5 else -1
        self.fr = 0
        self.t = 0
       
        
    def __init_subclass__(self, index = None):
        self.index=index                                                                    # I can't remember exactly why this was necessary
        
    def assignPosition(self,x,y):
        self.x =x                                                                           # assignPosition assigns the neuron x and y
        self.y=y
        self.position= (x,y)
    
    def pspWaveform(self,x):                                                                # returns the pspWaveform, given time value x
        return (np.e**(15*x*(self.psp_slow_decay-1)/self.psp_slow_decay) - np.e**(15*x*(self.psp_fast_decay-1)/self.psp_fast_decay)) * psp_potential
    
    # the step function is the main function of the Neuron, it takes an integer (time in milliseconds) and calculates its own inputs and runs through the izhikevich model in order to calculate a response
    def step(self, t):
        self.tune_input(curve = self.tuning_type)                                                               # Inputs to this neuron are stored as a list, this line takes the sum of all the potentials stored in the list to determine the total input
        self.__Izhikevich(t)                                                                # calls Izhikevich 
        self.I = self.rest_I.copy()  
        self.t += self.h                                                                      # redefines the input list to be [0] where future values will be appended before the next time step
        
    # Simple discrete model of the izhikevich solution for single neuronal dynamics
    def __Discrete_Model(self,a,b,u,v,I):
        v= v + self.h *(0.04*v*v+5*v+140-u+I)                                                   
        u = u + self.h *(a*(b*v-u))
        return u,v
    
    #Main Izhekivich function for solving single neuron dynamics
    def __Izhikevich(self, t):
        
        self.vMemory.append(self.v)                                                         # Stores the memory of the v value
        self.uMemory.append(self.u)                                                         # Stores the memory of the u value
        self.iMemory.append(self.I)                                                         # stores the memory of the sum of the inputs
        
        self.u,self.v=self.__Discrete_Model(self.a,self.b,self.u,self.v,self.I)             #Calls the discrete model for izhekivich

        if self.v > 30:                                                                     # checks if spike condition is met
            self.v = self.c
            self.u = self.u + self.d
            self.fire = True
            self.spikes.append(t)
            
        else:
            self.fire = False                                                               # resets neuron spikes boolean value
    
    
    def get_firing_rate(self):
        self.fr = len(self.spikes) / self.t 
        return self.fr
    
    def tune_input(self, curve = 'linear'):
        if curve == 'linear':
            self.I = sum(self.I)
        elif curve == 'square root':
            temp = self.tuning_direction*sum(self.I) + self.tuning_factor
            if temp > 0:
                self.I = np.sqrt(temp)
            else:
                self.I = 0
        elif curve == 'grid':
            self.I = sum(self.I)
        else:
            raise Exception("Unrecognized tuning type")
        
        
    def clear_mem(self):
        self.I = self.rest_I.copy()
        self.vMemory, self.iMemory, self.uMemory = [], [], []
        self.spikes = []
        self.t = 0
        
'''
NEURON OBJECT CHILDREN CLASSES

'''
    
    
    

    
#Each neuron type gets its own color
    
class regularSpiking(Neuron):
    
    def __init__(self, index = None, I = [0], tuning_type = 'linear'):
        Neuron.__init__(self)
        self.color = 'dodgerblue'
        self.cell_type = 'rs'
        self.index=index
        self.I = I
        self.a, self.b, self.c, self.d = parameters[self.cell_type]
        self.u, self.v = -65*self.b, -65
        self.psp_slow_decay, self.psp_fast_decay, self.psp_fast_potential, self.psp_slow_potential = PSP_parameters[self.cell_type]
        self.tuning_type = tuning_type
        
class fastSpiking(Neuron):
    
    def __init__(self, index = None, I = [0], tuning_type = 'linear'):
        Neuron.__init__(self)
        self.color = 'firebrick'
        self.cell_type = 'fs'
        self.index=index
        self.I = I
        self.a, self.b, self.c, self.d = parameters[self.cell_type]
        self.u, self.v = -65*self.b, -65
        self.psp_slow_decay, self.psp_fast_decay, self.psp_fast_potential, self.psp_slow_potential = PSP_parameters[self.cell_type]
        self.tuning_type = tuning_type
        
class lowThresholdSpiking(Neuron):
    def __init__(self, index = None, I = [0], tuning_type = 'linear'):
        Neuron.__init__(self)
        self.color = 'forestgreen'
        self.cell_type = 'lts'
        self.index=index
        self.I = I
        self.a, self.b, self.c, self.d = parameters[self.cell_type]
        self.u, self.v = -65*self.b, -65
        self.psp_slow_decay, self.psp_fast_decay, self.psp_fast_potential, self.psp_slow_potential = PSP_parameters[self.cell_type]
        self.tuning_type = tuning_type
class intrinsicBurst(Neuron):
    def __init__(self, index = None, I = [0], tuning_type = 'linear'):
        Neuron.__init__(self)
        self.color = 'yellow'
        self.cell_type = 'ib'
        self.index=index
        self.I = I
        self.a, self.b, self.c, self.d = parameters[self.cell_type]
        self.u, self.v = -65*self.b, -65
        self.psp_slow_decay, self.psp_fast_decay, self.psp_fast_potential, self.psp_slow_potential = PSP_parameters[self.cell_type]
        self.tuning_type = tuning_type
class dummyNode(Neuron):
    def __init__(self, index = None, I = [10], tuning_type = 'linear'):
        Neuron.__init__(self)
        self.color = 'orange'
        self.cell_type = 'dummy'
        self.index=index
        self.I = I
        self.rest_I = self.I
        self.a, self.b, self.c, self.d = parameters['rs']
        self.u, self.v = -65*self.b, -65
        self.psp_slow_decay, self.psp_fast_decay, self.psp_fast_potential, self.psp_slow_potential = PSP_parameters['rs']
        self.tuning_type = tuning_type

