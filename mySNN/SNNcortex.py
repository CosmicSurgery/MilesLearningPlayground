from SNNconnectivity import neuron_info_dict, connectivity, cellRange
from SNNneuron import regularSpiking, fastSpiking, lowThresholdSpiking, intrinsicBurst, dummyNode, Neuron, neuron_radius
from SNNsynapse import Synapse
from SNNnetwork import Network

import random
import numpy as np


class Cortex:
    
    def __init__(self,net = Network(), n=100,width = 1, height = 1, grid = False, layer = False, name = None, trim = False, tuning_type = 'linear', x=None, y=None):                                                                         
        self.n = n                                                                                      #Optional parameters define amount of neurons and synapses in the network
        self.net = net     
        self.width = width
        self.height = height
        self.neurons = []                                                                               #Neurons list stores all of the neuron objects   
        self.synapses = []         
        self.x, self.y = x, y                                                                     #synapses list stores all of the synapse objects
        if name is None:
            name = len(net.cortices)
        
        self.grid = grid
        self.name = name
        self.trim = trim
        self.layer = layer
        if self.layer:
            self.width = 0
            self.trim = False
            self.height = neuron_radius * 2 * self.n
        self.tuning_type = tuning_type
        if self.grid:
            self.__setupGrid(self.neurons)
        else:
            self.__setup()
    
    '''
    STARTUP FUNCTIONS
    __setup
        __generateNeurons
            __legalPlacement
                __randomNeuronType
                    __assign
        __generateSynapses
            __doesConnect
            __formConnections
        __trimCortex
        __random_fix
        __sync
            
    CORTEX OBJECT ATTRIBUTES
        n = number of neurons within cortex
        region = the region that cortex occupies in space
    
    
    __setup is a hidden function that gets called on initialization of a new cortex.
    A default cortex contains 100 neurons, and occupies a 1 by 1 mm region of space
    
    once called it loops through the given number of neurons in the cortex object to 
    
    
    '''
    
    def __setup(self):
        self.__generateNeurons()
        if not self.layer:
            self.__generateSynapses()
        if self.trim:
            self.__trimCortex()
        else: 
            self.__random_fix()
        self.__sync()
        self.net.cortices.append(self)
    
    def __setupGrid(self, neurons):
        for x in range(self.x):
            for y in range(self.y):
                self.neurons.append(regularSpiking(tuning_type = 'grid'))
                neurons[-1].assignPosition(*(x,y))
                neurons[-1].cortex = self.name
        self.__sync()
    
    def __generateNeurons(self):
        for j in range(self.n):
            while not self.__legalPlacement(j, self.neurons):            # Gives each newly geneated neuron a position
                pass
 
        # Gives a random position to a neuron and returns whether it is valid or not
    def __legalPlacement(self, j, neurons):
        (x,y) = random.uniform(self.net.width, self.net.width + self.width), random.uniform(0, self.height)
        for k in neurons:
            if np.linalg.norm(np.array(k.position)-np.array((x,y))) <= neuron_radius:
                return False
        neurons.append(self.__randomNeuronType(j))
        neurons[j].assignPosition(*(x,y))
        neurons[j].cortex = self.name
        return True
    
    #stores the indices of each neuron object in the neurons list according to neuron type in a special neuron type list
    def __randomNeuronType(self, j):
        cell_type = self.__assign()
        if cell_type == 'rs':
            return regularSpiking(index=j, tuning_type = self.tuning_type)
        elif cell_type == 'fs':
            return fastSpiking(index=j, tuning_type = self.tuning_type)
        elif cell_type == 'lts':
            return lowThresholdSpiking(index=j, tuning_type = self.tuning_type)
        elif cell_type == 'ib':
            return intrinsicBurst(index=j, tuning_type = self.tuning_type)
        else:
            raise Exception("Unrecognized Neuron Cell Type")
        
        
    
    
    
    #Assigns the neuron type according to neuron type distribution
    def __assign(self, excitatory = False):
        range_max = 100
        for cell_type in neuron_info_dict:
            if random.randrange(0,range_max) < neuron_info_dict[cell_type]['prolific']*100:
                return cell_type
            range_max -= neuron_info_dict[cell_type]['prolific']*100
    
    #generates random connections
    def __generateSynapses(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.__doesConnect(i,j) :
                    self.__formConnections(self.neurons[i],self.neurons[j])
                    self.synapses[-1].length = self.__get_distance(self.neurons[i], self.neurons[j])
    
    '''
    __doesConnect takes in a pre-synaptic neuron and a post-synaptic neuron and determines if they will form a synapse
    according to certain connecting probability distributions outlined in SNNconnectivity
    '''
    
    def __doesConnect(self, pre, post):
        distance_index = int(np.round(1000 * self.__get_distance(self.neurons[pre], self.neurons[post]))) 
        # If distance is out of range of the cellRange distribution it is set to the smallest probability in the distribution
        if distance_index >= 1000:
            distance_index = len(cellRange[self.neurons[pre].cell_type]) -1
        connection_probability = cellRange[self.neurons[pre].cell_type][distance_index] * connectivity[self.neurons[pre].cell_type][self.neurons[post].cell_type] # determines the likelihood of a connection forming between these two neurons
    
        if random.uniform(0,1) < connection_probability:       #randomly determines draws from the distribution
            return True
        else:
            return False
    
    # creates a synapse object and appends it to the synapse list
    def __formConnections(self,inp,out):
        self.synapses.append(Synapse(inp, out))
            
    def __get_distance(self, inp, out):
    	return np.linalg.norm(np.array((inp.position) - np.array(out.position)))    
    
    # randomly connect neurons that are either without an input or an output (alternative to trimming the cortex)
    def __random_fix(self):
        alone_neurons = [k for k in self.neurons if not (k.is_an_input and k.is_an_output)]
        for k in alone_neurons:
            new_neuron = k
            while new_neuron == k:
                new_neuron = self.neurons[random.randrange(0,len(self.neurons))] 
            if not k.is_an_input:
                self.__formConnections(k, new_neuron)
            elif not k.is_an_output:
                self.__formConnections(new_neuron, k)
    
    # remove all redundant synapses and neurons from the cortex
    def __trimCortex(self):
        changed = True
        while changed:
            changed = False
            for i,k in enumerate(self.neurons):
                if not(k.is_an_input and k.is_an_output):
                    del(self.neurons[i])
                    changed = True
            for i,k in enumerate(self.synapses):
                temp_condition = not(k.inp.is_an_output and k.out.is_an_input)
                if temp_condition:
                    changed = True
                    k.inp.is_an_input -= 1
                    k.out.is_an_output -= 1
                    del(self.synapses[i])           
    
    def __sync(self):
        self.net.width += self.width + self.net.cortex_buffer
        self.net.height = max(self.net.height, self.height)
        self.net.n += self.n
        self.net.s += len(self.synapses)
        self.net.neurons += self.neurons
        self.net.synapses += self.synapses
    '''
    Public Cortex functions
    '''
            

         
            


    
