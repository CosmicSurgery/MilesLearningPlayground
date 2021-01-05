total_time=10                                               # measured in milliseconds
fs = 10000                                                  # sampling rate - 10kHz
h = 0.5


class Network:
    
    def __init__(self,n=10,s=20):
        self.n=n
        self.s=s
    
    pass


class Neuron:
    
    def __init__(self,*streams_step,a=0.02,b=0.2,c=-65,d=8):
        self.streams_step = streams_step                    #an immutable tuple of input potentials
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.stream = np.ones(fs*total_time-1)                   #creates an 
        self.clk = 0                                             #Each neuron has its own internal clock starting at t = 0
        self.v = -65*np.ones(fs*total_time)               #Each neuron has a starting membrane potential
        self.u = 0*self.v[:]
    
    def __Discrete_Model(u,v,I):                        #private method
        v = v + 0.5 *(0.04*v*v+5*v+140-u+I) 
        u = u + 0.5 *(self.a*(self.b*v-u))
        return u,v
    
    def __step(self,i):
        self.u[clk+1],self.v[clk+1]=Discrete_Model(self.u[clk],self.v[clkk],self.I[clk])
        self.clk += 1
    
    def current_potential(self):
        return v(clk)
    
class Synapse:
    
    def __init__(self, preSynapse, postSynapse,weight =1): #axonal and dendritic neurons are taken as paramters respectively
        self.preSynapse = preSynapse
        self.postSynapse = postSynapse
        self.weight = weight
        
    def run():
        
        
a = Neuron()

