from SNNsynapse import Synapse

from operator import attrgetter
neuron_radius = 50 / 1000
fs = 10000               # Each time step is 0.1 millisecond
T1, T2, h=0, 100, 0.1

#A Network object composed of Neuron and Synapse objects
class Network:
    
    #Initializes the Network object
    def __init__(self):         
        self.n, self.s = 0, 0
        self.neurons = []                                                                               #Neurons list stores all of the neuron objects     
        (self.rs_id, self.fs_id, self.lts_id, self.ib_id) = [], [], [], []                              #Stores indicess to neurons of each neuron type
        self.synapses = []                                                                              #synapses list stores all of the synapse objects
        
        self.cortex_buffer = 0.5
        self.cortices = []
        self.width = 0
        self.height= 1
    
    '''
    takes an input neuron and an output neuron and creates a synapse and modifies neuron and synapse attributes
    '''
    
    def __formConnections(self,inp,out):
        
        self.synapses.append(Synapse(inp, out))
    
    '''
    takes two cortices and connects them accourding to parameters
    random_select - if random select is True the neurons will not be chosen based on their spatial position
    num_connections - specifies the number of synapses connecting the cortices
    '''
    
    def connectCortices(self,cortex_one, cortex_two, random_select = False, num_connections = 20):
    
        print(len(self.getOutputNeurons(cortex_one, random_select = random_select, num_connections = num_connections)))
        for i in range(num_connections):
            self.__formConnections(self.getOutputNeurons(cortex_one, random_select = random_select, num_connections = num_connections)[i],self.getInputNeurons(cortex_two, random_select = random_select, num_connections=num_connections)[i])
            
    def get_neurons_from(self,cortex):
        return [k for k in self.neurons if k.cortex == cortex]
    
    # cortex defines the name of the cortex that will check the display status
    # i will define whether the cortex neurons are checking for target or cursor position 0 for target and 1 for cursor
    def checkDisplay(self, cortex, i):
        for k in self.get_neurons_from(cortex):
            k.rest_I = [self.task[k.x, k.y][i]]
            
    
            
    #task is a numpy array of two dimensions where each element is an rbg dtype8 ie. [0,0,0] 'black' or [250, 0, 0] 'red'
    def setTask(self, task):
        self.task = task
    
    
    
    '''
    get OutputNeurons // getInputNeurons simply returns a list of neurons according to certain parameters
    cell_type - which cell types should be considered
    '''
    
    
    def getOutputNeurons(self, cortex, cell_type = ['rs', 'ib'], num_connections = 20, random_select = False): # defaults to choose only excitatory neurons
        return sorted([k for k in self.neurons if (attrgetter('cell_type')(k) in cell_type and attrgetter('cortex')(k) == cortex.name)], key=attrgetter('x'))[-num_connections:]
        pass
    
    def getInputNeurons(self, cortex, cell_type = ['rs', 'ib'], num_connections = 20, random_select = False): # defaults to choose only excitatory neurons
        return sorted([k for k in self.neurons if (attrgetter('cell_type')(k) in cell_type and attrgetter('cortex')(k) == cortex.name)], key=attrgetter('x'))[:num_connections]
        
    


