'''
Single Neuron 
Izhikevich model
regular spike (RS)
'''
#LIBRARY
# vector manipulation
import numpy as np
import sys

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# Input step-function
h = 0.5
input_onset=300
input_amp=5

time=np.arange(0,1000.1,h)

def Input(input_onset,input_amp):
    I=np.zeros(len(time))
    spikes = 5
    period = spikes / (100 * 2 * np.pi)
    
    for k in range (0,len(time)):
        if time[k] >input_onset:
            I[k]=input_amp #* np.sin(period * k)
    return I

# Numerical Solution to the Izhikevich model
def Discrete_Model(a,b,u,v,I):
    v= v + h *(0.04*v*v+5*v+140-u+I) 
    u = u + h *(a*(b*v-u))
    return u,v


def Izhikevich(a,b,c,d,title='temp'):
    v=-65*np.ones((len(time)))
    u=0*np.ones((len(time)))
    u[0]=b*v[0]

    I=Input(input_onset,input_amp)
    #EULER METHOD
    for k in range (0,len(time)-1):
        u[k+1],v[k+1]=Discrete_Model(a,b,u[k],v[k],I[k])
        
        if v[k+1]>30:
            v[k+1]=c
            u[k+1]=u[k+1]+d
    plot_input_output(time,v,I,a,b,c,d,title)
    
def plot_input_output(time,v,I,a,b,c,d,title):
    
    if title == 'temp':
        title = 'Parameters a %s b: %s c:  %s d:  %s' %(a,b,c,d)
    
    # PLOTTING
    fig, ax1 = plt.subplots(figsize=(12,3))
    ax1.plot(time, v, 'b-', label = 'Output')
    ax1.set_xlabel('time (ms)')
    # Make the y-axis label, ticks and tick labels match the line color.
    # Plotting out put 
    ax1.set_ylabel('Output mV', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim(-95,40)
    ax2 = ax1.twinx()
    # Plotting input on a different axis
    ax2.plot(time, I, 'r', label = 'Input')
    ax2.set_ylim(0,input_amp*20)
    ax2.set_ylabel('Input (mV)', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)
    ax1.set_title(title)
    f = 'images/'+title + '.png'
    plt.savefig(f,dpi=300)
    plt.show()

#Regular Spiking (RS) excitatory neuron
rs = (0.02,0.2,-65,8, "(RS) Neuron sin")

#Low-threshold (LTS) inhibitory neuron
lts = (0.02,0.25,-65,2, "(LTS) Neuron sin")

#fast spiking (FS) inhibitory neuron
fs = (0.1,0.2,-65,2, "(FS) Neuron sin")

#intrinsically bursting (IB) inhibitory neuron
ib = (0.02,0.2,-55,4, "(IB) Neuron")
'''
Izhikevich(*rs)
Izhikevich(*fs)
Izhikevich(*lts)
'''
Izhikevich(*ib)
