# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:07:05 2022

@author: kshah23
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Transaction Networks
transaction_network = pd.read_csv('../datasets/Transactions/transaction_data_normalized.csv')
trans_nodes = transaction_network.Source.unique()

i = 0
network_nodes = []
for tn in trans_nodes[i:i+8]:
    network_nodes.append(transaction_network[transaction_network['Source'] == tn])

# Trust Networks
trust_network = pd.read_csv('../datasets/interpolation/masterEdgeList.csv', header=None)
trust_network.columns = ['from','to','weight']
trust_edges = []
for i in range(1,8+1):
    tr_edge = pd.DataFrame(trust_network[trust_network['from']==i])
    tr_edge.columns = ['from','to','weight']
    for i,j in tr_edge.iterrows():  
        trust_edges.append((int(j['from']),int(j['to']),j['weight']*100))

G = nx.DiGraph()
G.add_weighted_edges_from(trust_edges, arrows=True)

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

options = {
    'node_color': 'orange',
    'node_size': 300,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
}
nx.draw(G, nx.circular_layout(G), with_labels = True, edge_color=weights, edge_cmap=plt.cm.Blues, arrows=True,  **options)
plt.show() # display


categories = set(transaction_network['Transaction_Type'])
default_attrs = {}
for cat in categories:
    default_attrs[cat] = 0
    
all_attrs = {}
for i in range(1,8+1):
    node_params = default_attrs
    all_attrs[i] = {}
    for cat in categories:
        node_params[cat] = network_nodes[i-1][[network_nodes[i-1]['Transaction_Type']=='food'][0]]['Weight'][i-1]
    all_attrs[i] = node_params

nx.set_node_attributes(G, all_attrs)

