
import networkx as nx
import matplotlib.pyplot as plt


class visual:

    def showRaster(net):
        fig = plt.figure()
        neuralData = []
        for n in net.neurons:
            neuralData.append(n.spikes)
        plt.eventplot(neuralData)
        plt.ylabel('Neuron')
        plt.xlabel('Spiketime in ms')
        plt.show()
        
    def showLayer(net, scheme = 'cell_type', labels_on = False):
        fig = plt.figure()
        ax = fig.add_subplot()
        G = nx.Graph()
        for i,n in enumerate(net.neurons):
            G.add_node(n,pos=n.position)
        color_map, labels = visual.__color_scheme(net, G, scheme=scheme)
        pos = nx.get_node_attributes(G,'pos')
        
        mpl = nx.draw_networkx_nodes(G,pos, ax=ax, node_color=color_map, node_cmap = plt.cm.Oranges, node_size=net.neurons[0].radius*5000)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map, node_cmap = plt.cm.Oranges, node_size=net.neurons[0].radius*5000)
        if labels_on:
            nx.draw_networkx_labels(G,pos,labels)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.title(scheme)
        plt.colorbar(mpl)
        plt.show()
        
    def debug(net,scheme = 'cell_type'):
        fig = plt.figure()
        ax = fig.add_subplot()
        G = nx.Graph()
        color_map = []
        labels = {}
        for i,n in enumerate(net.neurons):
            G.add_node(net.neurons[i],pos=net.neurons[i].position)
        color_map, labels = visual.__color_scheme(net, G, scheme=scheme)
        
        
        pos = nx.get_node_attributes(G,'pos')
        mpl = nx.draw_networkx_nodes(G,pos, ax=ax, node_color=color_map, node_cmap = plt.cm.Oranges, node_size=net.neurons[0].radius*5000)
        
        nx.draw_networkx_labels(G,pos,labels)
        plt.colorbar(mpl)
        plt.title(scheme)
        plt.show()


    def showNetwork(net, scheme= 'cell_type',labels_on = False):
        if len(net.synapses) <= 0:
            raise Exception("Network has no synapses")
        fig = plt.figure()
        ax = fig.add_subplot()
        G = nx.Graph()
        color_map = []
        labels = {}
        for i,n in enumerate(net.neurons):
            G.add_node(net.neurons[i],pos=net.neurons[i].position)
        color_map, labels = visual.__color_scheme(net, G, scheme=scheme)
        for s in net.synapses:
            G.add_weighted_edges_from([(s.inp,s.out,s.weight)])
            
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        
        pos = nx.get_node_attributes(G,'pos')
        mpl = nx.draw_networkx_nodes(G,pos, ax=ax, node_color=color_map, node_cmap = plt.cm.Oranges, node_size=net.neurons[0].radius*5000)
        nx.draw(G, pos, ax=ax, edge_color=weights, edge_cmap=plt.cm.Blues, alpha=0.7)
        if labels_on:
            nx.draw_networkx_labels(G,pos,labels)
        limits=plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.colorbar(mpl)
        plt.show()

    def __color_scheme(net, G, scheme='cell_type'):
        color_map = []
        labels = {}
        if scheme == 'cell_type':
            for i,node in enumerate(G):
                color_map.append(net.neurons[i].color)
        elif scheme == 'potential':
            for i,node in enumerate(G):
                color_map.append(1000* sum(net.neurons[i].vMemory)/len(net.neurons[i].vMemory))
                labels[node] = int(color_map[-1])
        elif scheme == 'debug':
            for i,node in enumerate(G):
                color_map.append(net.neurons[i].debug)
        elif scheme == 'frequency':
            for i,node in enumerate(G):
                color_map.append(1000 * net.neurons[i].get_firing_rate())
                labels[node] = int(color_map[-1])
        return color_map, labels


    '''

    BELOW IS BUGGED

    '''












    '''


        #Plots the neurons and their connections
    def visualize(n):   #Takes in a network of neurons and synapses
        fig = plt.figure()
        
        if True:
            ax = fig.add_subplot()
            visual.__showNetwork(ax)
        if False:
            ax = fig.add_subplot(122)
            visual.__neuronTypeDistribution(ax)
        toggle = False
        if toggle:
            ax = plt.subplots(111)
            visual.__connectionDistribution(ax)
            (ax1,ax2) = plt.subplots(2)
            __showNetwork(ax1)
            __neuronTypeDistribution(ax2)
        plt.show()

    def __showNetwork(ax):
        count=0
        G = nx.Graph()
        color_map = []
        for i,n in enumerate(n.neurons):
            G.add_node(n.neurons[i],pos=n.neurons[i].position)
        color_map = __color_scheme(net, G)
        for s in n.synapses:
            G.add_weighted_edges_from([(s.inp,s.out,s.weight)])
            
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, ax=ax,node_color=color_map, edge_color=weights, edge_cmap=plt.cm.Blues, alpha=0.7,node_size=neuron_radius*5000)#with_labels=True)
        limits=plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.legend()
        

        
    def __connectionDistribution(ax):
        dist = [s.length for s in n.synapses]
        ax.hist(dist,color='orange',alpha=0.5)
        
    def __neuronTypeDistribution(ax):
        sizes = [len(n.rs_id),len(n.fs_id),len(n.lts_id),len(n.ib_id)]
        labels = ['rs','fs','lts','ib']
        ax.pie(sizes, labels=labels,autopct='%1.1f%%',colors=['dodgerblue','firebrick','green','yellow'])
        
    '''
            
