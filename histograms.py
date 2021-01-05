"""
Single Neuron Spiking Dynamics with Izhikevich Model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.signal import find_peaks
from scipy.stats import pearsonr

def genInput():
    inputSignal = np.zeros((int(total_time + step)*sf))
    inputEvents = genEvents()
    for x in inputEvents:
        inputSignal = psp(inputSignal,(x*sf))
    return inputSignal
    
def genEvents():
    inputEvents = np.random.poisson(psp_freq, total_time)
    inputEvents = [1000 / x for x in inputEvents] 
    inputEvents = np.cumsum(inputEvents)
    return [x for x in inputEvents if x < total_time]

def pspWaveformFixed(x):
    return np.e**(15*x*(psp_slow_decay-1)/psp_slow_decay) - np.e**(15*x*(psp_fast_decay-1)/psp_fast_decay)

def pspWaveform(x):
    return np.e**(-1*(x/psp_width_scalar)*(psp_fast_decay)) - np.e**(-1*psp_width_scalar*x*(psp_slow_decay))

#setting a psp to last 20 ms
def psp(signal,start=0):
    start = int(start)
    time_limit = len(signal)
    signal = np.concatenate((signal,np.zeros(psp_length*sf)))
    psp = np.arange(0,psp_length,step)  
    for i in range(len(psp)):
            signal[i+start] += pspWaveformFixed(psp[i])*psp_potential
    return signal[0:time_limit]



#plots raw signal in subplot row 1 and a histogram of the ISI in row 2
def analyze(inputSignal):
    time = np.arange(0,len(sig)/sf,step)
    peaks, _ = find_peaks(inputSignal) 
    inputIntervals = np.diff(np.asarray(peaks))
    inputIntervals = [int(x/sf) for x in inputIntervals]
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(time,inputSignal)
    peaks, _ = find_peaks(inputSignal,height=-40) 
    inputIntervals = np.diff(np.asarray(peaks))
    inputIntervals = [int(x/sf) for x in inputIntervals]
    ax2.hist(inputIntervals,bins)
    #finds peaks in milliseconds
    ax1.set_title("Signal")
    ax1.set(xlabel='time (ms)', ylabel='Potential (mV)')
    anchored_info = AnchoredText(str(len(peaks))+" spikes \n"+str(len(peaks)* 1000/total_time)+" Hz",loc=2)
    ax1.add_artist(anchored_info)
    ax2.set(xlabel='deltaT (ms)', ylabel='occcurances')
    ax2.set_title("ISI Histogram")
    ax1.plot([int(x/sf) for x in peaks], inputSignal[peaks], "x")
    #plt.show()

#plots overlapping histograms of the ISI's of two signals
def ISIhist(s1, s2):
    time = np.arange(0,len(s1)/sf,step)
    peaks1, _ = find_peaks(s1) 
    inputIntervals1 = np.diff(np.asarray(peaks1))
    inputIntervals1 = [int(x/sf) for x in inputIntervals1]
    peaks2, _ = find_peaks(s2,height=-40)
    inputIntervals2 = np.diff(np.asarray(peaks2))
    inputIntervals2 = [int(x/sf) for x in inputIntervals2]
    (counts1, _,_) = plt.hist(inputIntervals1,bins,alpha=0.5, label="input")
    (counts2, _,_) = plt.hist(inputIntervals2,bins,alpha=0.5, label="response")
    
    plt.text(psp_freq,20,"Input freq = %s\nOutput freq = %s\npsp_length = %s\npsp_amplitude = %s\nslow_decay coefficient = %s" % (
        len(peaks1)*1000/total_time, len(peaks2)*1000/total_time,psp_length, psp_potential*scaling_factor,psp_slow_decay))
    plt.xlabel('dT (ms)')
    plt.ylabel('occurances')
    plt.title('ISI histogram of %s neuron at input rate %s Hz for %s ms' % (select_neuron,psp_freq, total_time))
    plt.legend(loc='upper right')
    
    #plt.show()
    #plt.savefig('test.png',dpi=300)
    corr, _ = pearsonr(counts1,counts2)
    return corr
    
def Discrete_Model(a,b,u,v,I):
    v= v + h *(0.04*v*v+5*v+140-u+I) 
    u = u + h *(a*(b*v-u))
    return u,v

def IzhikevichResponse(a=0.02,b=0.2,c=-65,d=8,I=5):
    v= -65*np.ones(len(I))
    u= 0*v[:]
    u[0]=b*v[0]
    
    for k in range (0,len(I)-1):
        u[k+1],v[k+1]=Discrete_Model(a,b,u[k],v[k],I[k])
        
        if v[k+1]>30:
            v[k+1]=c
            u[k+1]=u[k+1]+d
    return v

def displaymatrix():
    #plots redundant features from ISIHist method in first pop-up window
    for i in range(10):
        for j in range(10):
            psp_width_scalar=datPSPwidth[j]
            psp_potential=datPSPamp[i]
            sig=genInput()
            response = IzhikevichResponse(*neurons[select_neuron], sig)
            pspmatrix[i,j] = np.nan_to_num(ISIhist(sig, response))
            
    # pspmatrix where the first index is amplitude and second is width
    print(pspmatrix)
    plt.show()
    plt.imshow(pspmatrix)
    plt.colorbar()
    plt.show()

total_time = 100   # milliseconds
sf = 10             # khz
step = 0.1          # milliseconds
h = 0.5
psp_length = 20     # milliseconds
psp_freq = 45       # Hz
psp_potential = 40   # millivolts
psp_fast_decay = 7/8
psp_slow_decay = 31/32
neurons = {}
select_neuron = 'rs'
scaling_factor=0.037427605592999992         #y-value at psp-max
psp_potential /= scaling_factor
psp_width_scalar = 1
bins = np.arange(0,41,1)
#Izhikevich Conditions
#Regular Spiking (RS) excitatory neuron
neurons['rs'] = (0.02,0.2,-65,8)
#Low-threshold (LTS) inhibitory neuron
neurons['lts'] = (0.02,0.25,-65,2)
#fast spiking (FS) inhibitory neuron
neurons['fs'] = (0.1,0.2,-65,2)
#intrinsically bursting (IB) inhibitory neuron
neurons['ib'] = (0.02,0.2,-55,4)
#END OF INTIALIZE
sig = genInput()
'''
scaling_factor = np.floor((-np.log(np.log(psp_fast_decay)/np.log(psp_slow_decay)))/(np.log(psp_fast_decay)-np.log(psp_slow_decay)))
scaling_factor = (psp_slow_decay**scaling_factor - psp_fast_decay**scaling_factor)**(-1)
psp_potential = np.ceil(psp_potential*scaling_factor)
'''

#testing set
datFreq = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
freqCorr = np.zeros(len(datFreq))
datPSPamp = [2,3,4,5,6,7,8,9,10,11]
datPSPamp = [x/scaling_factor for x in datPSPamp]
datPSPwidth = [0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25]
pspmatrix = np.zeros((10,10))
#Ideal parameters according to the cross correlation are amplitude = 4 and width = 1


time = np.arange(0,len(sig)/sf,step)

response = IzhikevichResponse(*neurons[select_neuron], sig)

analyze(sig)
#analyze(response)
plt.show()
#plt.plot(time,response)


'''
plt.show()
plt.plot(freqCorr,label='Freq' + str(datFreq))
plt.plot(ampCorr,label='amplitude'+str(np.floor(datPSPamp)))
plt.plot(widthCorr,label='PSPwidth'+str(datPSPwidth))
plt.legend(loc=1)
plt.show()

sig = genInput(psp_potential,total_time,psp_freq)
response = IzhikevichResponse(*neurons[select_neuron], sig)

time = np.arange(0,len(sig)/sf,step)

analyze(response)

#ISIhist(input, response)
'''
