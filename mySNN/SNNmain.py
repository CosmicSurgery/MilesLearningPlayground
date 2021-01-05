'''
from SNNnetwork import Network
net = Network()

from SNNcortex import Cortex
P = Cortex(net, grid = True, x=8, y=8, name = 'P')



from SNNdisplay import display as d
data, disp_img, img = d.display()

net.setTask(data)

net.checkDisplay('P', 0)

T1, T2, h = 0, 100, 0.1

import numpy as np
clk = np.arange(T1,T2,h)
for t in clk:
    for n in net.neurons:
        n.step(t)
disp_img.show()
from SNNvisual import visual as v
v.showLayer(P, scheme = 'frequency')
v.showRaster(net)
'''
from SNNdemos import demos

demos = demos()
demos.one_in_one_out()




'''
TODO

-fix psps in synapse and neuron
-make a flow chart to help make the code easier to read

-


-fix conduction delay in Synapse() and define Synapse lengths               DONE
-fix problem with islands and peninsulas in the cortices.                   DONE
-filter intermdiary connection neurons to only be excitatory                DONE
-define weights within synapses                                             DONE
-check that the neural network cann run with 2 cortices
-function that maps neurons that fire most frequently visually (colormap)   
-function that maps nueron average potentials (colormap)                    

-outline a better organizational structure for neurons and synapses and other data
    ->Currently some data stored in cortex level, and some data stored in network level
    ->Solution: all network level structures that have a cortex duplicate must be pointers to the cortex level structure?

'''
