import numpy as np
import random

from SNNconnectivity import noise, sf

#Synapse class stores relevant info
class Synapse:
    
    def __init__(self, inp, out): 
        self.inp = inp                                                                         # Stores a reference to the input Neuron object
        self.out = out                                                                         # Stores a reference to the output Neuron object
        self.clk = 0                                                                            # clk used for debugging
        self.h = 0.1                                                                            # step size( milliseconds)
        self.offset = 0                                                                         # offset to set base and reset potential that the synapse will pass to the output neuron
        self.delay = 1        # 1 millisecond                                        # conduction delay value in milliseconds (default set to 1 ms)
        self.fast_memory, self.slow_memory, self.net_memory, self.conduction_delay = [0], [0], [], [0]        # memory stores all of the previous potential values that the synapse has sent
        self.length = None                                                                      # length of the synapse
        self.weight = random.random()                                                           # random connection weight
        self.psp_slow_decay, self.psp_fast_decay, self.psp_slow_potential, self.psp_fast_potential = self.inp.psp_slow_decay, self.inp.psp_fast_decay, self.inp.psp_slow_potential, self.inp.psp_fast_potential
        self.scaling_factor = np.floor((-np.log(np.log(self.psp_fast_decay)/np.log(self.psp_slow_decay)))/(np.log(self.psp_fast_decay)-np.log(self.psp_slow_decay)))
        self.scaling_factor = (self.psp_slow_decay**self.scaling_factor - self.psp_fast_decay**self.scaling_factor)**(-1)
        self.__setup()
        
    # This definition is responsible for changing the relevant attributes of the neurons it is connected to
    def __setup(self):
        self.inp.is_an_input += 1
        self.out.is_an_output += 1
        self.conduction_delay *= int(self.delay * sf)
        if None not in self.inp.position  and None not in self.out.position:
            self.length = np.linalg.norm(np.array(self.inp.position)-np.array(self.out.position))
        
    def set_delay(self, delay):
        self.delay = delay
        self.conduction_delay = [0] * int(self.delay *sf)
        # FOR DEBUGGING used to reset all future potentials in a synapse to new offset values 
    def reset(self):
        self.slow_memory[-1], self.fast_memory[-1] = 0 , 0
        
        # main function in the synapse class for pushing potential through the synapse
    def step(self, t):
        if self.inp.cell_type != 'dummy':
            spontaneous_noise = np.random.uniform(-noise, noise)
            if self.inp.fire == True:                                                               # checks to see if the input neuron is currently firing
                self.slow_memory.append(self.slow_memory[-1] + self.psp_slow_potential)
                self.fast_memory.append(self.fast_memory[-1] + self.psp_fast_potential)
            else:
                self.slow_memory.append(self.slow_memory[-1] * self.psp_slow_decay)
                self.fast_memory.append(self.fast_memory[-1] * self.psp_fast_decay)
            self.conduction_delay.append((self.slow_memory[-1] - self.fast_memory[-1])*self.scaling_factor + spontaneous_noise)    
        else:
            self.conduction_delay += self.inp.rest_I
        self.net_memory.append(self.conduction_delay[0])							# add the first element of conduction_delay to net_memory
        del(self.conduction_delay[0])										# delete the first element of conduction_delay			
        self.out.I.append(self.net_memory[-1])								# add the most previous first elemtnt of conduction delay to the out neurons I list
        self.out.debug +=1											
		
        


