import queue
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#Miles' Package that outlines the SNN connection parameters
from SNNconnectivity import neuron_info_dict
from SNNconnectivity import connectivity
from SNNconnectivity import cellRange

neuron_radius = 50          #radius of the neuron in microns
neuron_radius /= 1000

neuron_density_per_mm = 1 / (np.pi * neuron_radius**2)      #79,500 units max can fit in a 1mm space (can scale size of grid to units?)

debug = [[] for i in range(100)]
debug2 = [0]
debug3 = [0]

#A Network object composed of Neuron and Synapse objects
class Network:
    
    #Initializes the Network object
    def __init__(self,n=100,region =[(0,1),(0,1)]):                                                                         
        self.n = n                                                                         #Optional parameters define amount of neurons and synapses in the network                       
        self.region = region
        self.neurons = []                                                                               #Neurons list stores all of the neuron objects     
        (self.rs_id, self.fs_id, self.lts_id, self.ib_id) = [], [], [], []                              #Stores indicess to neurons of each neuron type
        self.axonTerminals, self.dendriticSpines = [], []                                               #Data structure not used yet
        self.synapses = []                                                                              #synapses list stores all of the synapse objects
        self.activeConnections = []         # stores all tuples of all connections
        self.construct()
        
    #String to display when this object is printed
    def __str__(self):
        return "There are %s neurons and %s synpases in this network" % (self.n,self.s)
    
    #constructs the neural network with neuron and synapse objects
    def construct(self):
        for i in range(self.n):
            while not self.__legalPlacement(i):            # Gives each newly geneated neuron a position
                pass
        self.__generateConnections()
        
            #stores the indices of each neuron object in the neurons list according to neuron type in a special neuron type list
    def __generateRandomNeuron(self, i):
        cell_type = self.__assign()
        if cell_type == 'rs':
            self.rs_id.append(i)
            return regularSpiking()
        elif cell_type == 'fs':
            self.fs_id.append(i)
            return fastSpiking()
        elif cell_type == 'lts':
            self.lts_id.append(i)
            return lowThresholdSpiking()
        elif cell_type == 'ib':
            self.ib_id.append(i)
            return intrinsicBurst()
        else:
            raise Exception("Unrecognized Neuron Cell Type")
        
    # Gives a random position to a neuron and returns whether it is valid or not
    def __legalPlacement(self, i):
        (x,y) = random.uniform(*self.region[0]), random.uniform(*self.region[1])
        for j in self.neurons:
            if np.linalg.norm(np.array(j.position)-np.array(((x,y)))) <= neuron_radius:
                return False
            
        self.neurons.append(self.__generateRandomNeuron(i))
        self.neurons[i].position = ((x,y))
        return True
        
    #generates random connections
    def __generateConnections(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.__doesConnect(i,j) :
                    self.activeConnections.append((i,j))
                    self.synapses.append(Synapse())
                    self.synapses[-1].length = np.linalg.norm(np.array(self.neurons[i].position) - np.array(self.neurons[j].position))

    def __doesConnect(self, pre, post):
        distance_index = int(np.round(1000 * np.linalg.norm(np.array(self.neurons[pre].position) - np.array(self.neurons[post].position))))        #Converts the distance between two 
        if distance_index >= 1000:
            distance_index = len(cellRange[self.neurons[pre].cell_type]) -1
        connection_probability = cellRange[self.neurons[pre].cell_type][distance_index] * connectivity[self.neurons[pre].cell_type][self.neurons[post].cell_type]
        
        if random.uniform(0,1) < connection_probability:       #will give a percentage likelihood of their being a connection
            return True
        else:
            return False
        
        #Assigns the neuron type according to neuron type distribution
    def __assign(self):
        range_max = 100
        for cell_type in neuron_info_dict:
            if random.randrange(0,range_max) < neuron_info_dict[cell_type]['prolific']*100:
                return cell_type
            range_max -= neuron_info_dict[cell_type]['prolific']*100

        #Plots the neurons and their connections
    def visualize(self):
        fig = plt.figure()
        
        if True:
            ax = fig.add_subplot(121)
            self.__showNetwork(ax)
        if True:
            ax = fig.add_subplot(122)
            self.__neuronTypeDistribution(ax)
        toggle = False
        if toggle:
            ax = plt.subplots(111)
            self.__connectionDistribution(ax)
            (ax1,ax2) = plt.subplots(2)
            self.__showNetwork(ax1)
            self.__neuronTypeDistribution(ax2)
        plt.show()
    
    def __showNetwork(self,ax):
        G = nx.Graph()
        color_map = []
        for i,_ in enumerate(self.neurons):
            G.add_node(i,pos=self.neurons[i].position)
        for tup in self.activeConnections:
            G.add_edges_from([tup])
        for node in G:
            color_map.append(self.neurons[node].color)
            
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, ax=ax,node_color=color_map, alpha=0.7,node_size=neuron_radius*5000)#with_labels=True)
        limits=plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.legend()
        
    def __connectionDistribution(self,ax):
        dist = [s.length for s in self.synapses]
        ax.hist(dist,color='orange',alpha=0.5)
        
    def __neuronTypeDistribution(self,ax):
        sizes = [len(self.rs_id),len(self.fs_id),len(self.lts_id),len(self.ib_id)]
        labels = ['rs','fs','lts','ib']
        ax.pie(sizes, labels=labels,autopct='%1.1f%%',colors=['dodgerblue','firebrick','green','yellow'])
            

    #Completely empty neuron class
class Neuron:
    
    def __init__(self):
        self.cell_type = None
        self.position = ((None, None))
        self.radius = neuron_radius
        pass
    
    def set_input(self):
        pass
        
    def test_fun(self):
        pass
    
#Each neuron type gets its own color
    
class regularSpiking(Neuron):
    
    def __init__(self):
        Neuron.__init__(self)
        self.color = 'dodgerblue'
        self.cell_type = 'rs'

class fastSpiking(Neuron):
    def __init__(self):
        Neuron.__init__(self)
        self.color = 'firebrick'
        self.cell_type = 'fs'

class lowThresholdSpiking(Neuron):
    def __init__(self):
        Neuron.__init__(self)
        self.color = 'forestgreen'
        self.cell_type = 'lts'

class intrinsicBurst(Neuron):
    def __init__(self):
        Neuron.__init__(self)
        self.color = 'yellow'
        self.cell_type = 'ib'

#Synapse class stores relevant info
class Synapse:
    
    def __init__(self): 
        self.conduction= queue.Queue(10)                                          #propogation delay of 10 samples (or 1ms to travel the axon)
        for i in range(10):
            self.conduction.put(-65)
        self.preSynapse = None
        self.postSynapse = None
        self.pre_index = None
        self.post_index = None
        self.length = None
    
    def set_connections(self, preSynapse, postSynapse):
        self.preSynapse = preSynapse
        self.postSynapse = postSynapse

def display(networks):  #networks is a list of networks
    fig = plt.figure()
    
    ax = fig.add_subplot
    
    G = nx.Graph()
    color_map = []
    all_neurons  = networks[0].neurons + networks[1].neurons
    for i,_ in enumerate(


S1 = Network()
S1.visualize()
M1 = Network(region =[(1.5,2.5),(0,1)])
M1.visualize()
output_layer = []



 def visualize(self):
        fig = plt.figure()
        
        if True:
            ax = fig.add_subplot(121)
            self.__showNetwork(ax)
        if True:
            ax = fig.add_subplot(122)
            self.__neuronTypeDistribution(ax)
        toggle = False
        if toggle:
            ax = plt.subplots(111)
            self.__connectionDistribution(ax)
            (ax1,ax2) = plt.subplots(2)
            self.__showNetwork(ax1)
            self.__neuronTypeDistribution(ax2)
        plt.show()
    
    def __showNetwork(self,ax):
        G = nx.Graph()
        color_map = []
        for i,_ in enumerate(self.neurons):
            G.add_node(i,pos=self.neurons[i].position)
        for tup in self.activeConnections:
            G.add_edges_from([tup])
        for node in G:
            color_map.append(self.neurons[node].color)
            
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, ax=ax,node_color=color_map, alpha=0.7,node_size=neuron_radius*5000)#with_labels=True)
        limits=plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.legend()
        
    def __connectionDistribution(self,ax):
        dist = [s.length for s in self.synapses]
        ax.hist(dist,color='orange',alpha=0.5)
        
    def __neuronTypeDistribution(self,ax):
        sizes = [len(self.rs_id),len(self.fs_id),len(self.lts_id),len(self.ib_id)]
        labels = ['rs','fs','lts','ib']
        ax.pie(sizes, labels=labels,autopct='%1.1f%%',colors=['dodgerblue','firebrick','green','yellow'])
            























