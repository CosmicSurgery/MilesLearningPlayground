import numpy as np
import matplotlib.pyplot as plt
# my packages
from SNNnetwork import Network
from SNNcortex import Cortex
from SNNneuron import regularSpiking, fastSpiking, lowThresholdSpiking, intrinsicBurst, dummyNode, Neuron, neuron_radius
from SNNsynapse import Synapse
from SNNvisual import visual as v

'''
each demo function should take in a network, and run a simple demo 

run_full does not include any default visualization built-in to the function
'''

class demos:

    def my_demo(self, net = Network()):
        net = Network()
        P = Cortex(net,n=20, layer = True)
        S = Cortex(net)
        M = Cortex(net)

        net.connectCortices(P,S)
        net.connectCortices(S,M)


        T1, T2, h=0, 100, 0.1
        clk = np.arange(T1,T2,h)

        for t in clk:
            for n in net.neurons:
                n.step(t)
            for s in net.synapses:  
                s.step(t)
                
                
        v.showNetwork(net, scheme = 'frequency')
        v.showRaster(net)

    def run_full(self, net = Network()):
        T1, T2, h=0, 100, 0.1
        clk = np.arange(T1,T2,h)
        #inputs = self.cortices[0].getInputNeurons()
        for t in clk:
            print(t)
            for n in net.neurons:
                n.step(t)
            for s in net.synapses:
                s.step(t)
                
        plt.figure()
        v.showNetwork(net)
        v.showRaster(net)

    def one_in_one_out(self):
        T1, T2, h = 0, 100, 0.1
        clk = np.arange(T1, T2, h)  
        in1 = dummyNode()
        out1 = regularSpiking()
        s = Synapse(in1, out1)
        in1.rest_I = [6]

        for t in clk:
            in1.step(t)
            out1.step(t)
            s.step(t)

        plt.figure
        plt.plot(clk,in1.iMemory,label = 'offset=6')
        plt.plot(clk,in1.vMemory,label='neuron1')
        plt.plot(clk,s.net_memory,label='postsynaptic potential')
        plt.plot(clk,out1.vMemory,label='neuron2')
        plt.legend()
        plt.show()
        
    def two_in_one_out(self, net,delay=5):
        T1,T2, h = 0, 1000, 0.1
        clk = np.arange(T1,T2,h)
        in1 = dummyNode()
        in2 = dummyNode()
        in1.static_input, in2.static_input = True, True
        out1 = regularSpiking()
        s1 = Synapse(in1, out1)
        s2 = Synapse(in2, out1)
        s2.set_delay(5)
        for t in clk:
            in1.step(t)
            in2.step(t)
            s1.step(t),s2.step(t)
            out1.step(t)
        
        plt.figure
        plt.plot(clk,in1.vMemory,alpha=0.5,label='in')
        plt.plot(clk,out1.vMemory,label='out')   
        #plt.plot(clk,s1.net_memory,alpha=0.5)   
        #plt.plot(clk,s2.net_memory,alpha=0.5)   
        plt.plot(clk, out1.iMemory,alpha=0.5)
        plt.legend()
        plt.show()
