# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:07:05 2022

@author: kshah23
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import random

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


def main():
    categoryValue = 'barsandrestaurants'
    
    if categoryValue == "":
        print("Please enter category value as input while running the file")
        return
    
    # Trust Networks
    trust_network = pd.read_csv('../datasets/interpolation/masterEdgeList.csv', header=None)
    trust_network.columns = ['from','to','weight']

    # Transaction Networks
    transaction_network = pd.read_csv('../datasets/Transactions/transaction_data_normalized2.csv')
    trans_nodes = transaction_network.Source.unique()

    ma = max(max(trust_network['from']), max(trust_network['to']))
    trans_numbers = []

    trans_numbers = random.sample(range(1, len(trans_nodes)), ma)
    # for ite in range(1, ma+1):
    #     random_number = random.randrange(1,len(trans_nodes))
    #     while(random_number in trans_numbers):
    #         random_number = random.randrange(1,len(trans_nodes))
    #     trans_numbers.append(random_number)
    # print(len(trans_numbers))
    # print(trans_numbers)

    new_trans_nodes = []
    for trans_number in trans_numbers:
        new_trans_nodes.append(trans_nodes[trans_number])

    i = 0
    network_nodes = []
    for tn in new_trans_nodes:
        network_nodes.append(transaction_network[transaction_network['Source'] == tn])

    trust_edges = []
    for i in range(1,ma+1):
        tr_edge = pd.DataFrame(trust_network[trust_network['from']==i])
        tr_edge.columns = ['from','to','weight']
        for i,j in tr_edge.iterrows():  
            trust_edges.append((int(j['from']),int(j['to']),j['weight']))

    G = nx.DiGraph()
    G.add_weighted_edges_from(trust_edges, arrows=True)

    categories = set(transaction_network['Transaction_Type'])
    print(categories)
    default_attrs = {}
    for cat in categories:
        default_attrs[cat] = 0
        
    all_attrs = {}

    for i in range(len(trans_numbers)):
        node_params = default_attrs.copy()
        all_attrs[i+1] = {}

        for cat in set(network_nodes[i]['Transaction_Type']):
            node_params[cat] = np.array(network_nodes[i][network_nodes[i]['Transaction_Type']==cat]['Weight'])[0]
            #node_params[cat] = np.array(network_nodes[i-1][[network_nodes[i-1]['Transaction_Type']==cat]]['Weight'])[0]
        all_attrs[i+1] = node_params


    nx.set_node_attributes(G, all_attrs)

    # These are the influential nodes identified before multiple iterations started
    nodelist = getMostInfluentialNode(G, categoryValue)
    
    influentialSubgraph = G.subgraph(nodelist)
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_before_h1.png")
    
    # perform hypothesis 1
    
    # catValues = nx.get_node_attributes(G, categoryValue)
    # print("before updation of hypothesis 1 for nodes", catValues)
    
    hypothesis1(G, categoryValue)
    influentialSubgraph = G.subgraph(nodelist).copy()
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_after_h1.png")
    
    # catValues = nx.get_node_attributes(G, categoryValue)
    # print("after updation the category value for nodes", catValues)


    # perform hypothesis 2
    #hypothesis2(influentialSubgraph, categoryValue)
    
    # perform hypothesis 3
    hypothesis3(influentialSubgraph, categoryValue)
    return    
    


def hypothesis3(influentialSubgraph, categoryValue):
    print('In H3')
    nodes = list(influentialSubgraph.nodes())
    #print(nodes)
    G1 = influentialSubgraph.copy()
    
    edges_added = []
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)-1):
            source = nodes[i]
            target = nodes[j]
            if influentialSubgraph.has_edge(source, target) == False and influentialSubgraph.has_edge(target, source) == False:
                # create an edge between these 2 nodes with a default w
                influentialSubgraph.add_edge(source, target, weight = 0.0001)
                edges_added.append((source, target))
    
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_before_h3.png")
    hypothesis1(influentialSubgraph, categoryValue)
    G2 = influentialSubgraph.copy()
    
    
    for edges in edges_added:
        G2.remove_edge(edges[0], edges[1])
    
    print(nx.adjacency_matrix(G2)-nx.adjacency_matrix(G1))
    similarity = deltacon(G1, G2)
    print("Dissimilarity: "+str(1-similarity))
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_after_h3.png")
    
    
def hypothesis2(influentialSubgraph, categoryValue):
    print('In H2')
    nodes = list(influentialSubgraph.nodes())
    G1 = influentialSubgraph.copy()
    for i in range(len(nodes)-1):
        source = nodes[i]
        target = nodes[i+1]
        if influentialSubgraph.has_edge(source, target) == False and influentialSubgraph.has_edge(target, source) == False:
            # create an edge between these 2 nodes with a default w
            influentialSubgraph.add_edge(source, target, weight = 0.0001)
    
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_before_h2.png")
    hypothesis1(influentialSubgraph, categoryValue)
    G2 = influentialSubgraph.copy()
    #print(G2)
    
    similarity = deltacon(G1, G2)
    print(similarity)
    visualizeNetworkXGraph(influentialSubgraph, "graph_canvas_after_h2.png")

def visualizeNetworkXGraph(G, fileName):
    options = {
    'node_color': 'orange',
    'node_size': 300,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    }
    edges,weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, nx.circular_layout(G), with_labels = True, edge_color=weights, edge_cmap=plt.cm.Blues, 
    arrows=True,  **options)
    plt.savefig("../assets/" + fileName)
    plt.clf()

def hypothesis1(G, categoryValue):
    # step 1 : run 100 iterations over the graph
    # step 2 : for each iteration, get the most influential set of start points for the graph
    # step 3 : for each of that node, update the knowledge of the node and the edge weight also

    knowledgeImpactList = [-1, 0, 1]
    for i in range (0, 10):
        # print("Processing iteration ", i)
        mostInfluentialNodes = getMostInfluentialNode(G, categoryValue)
        # print(mostInfluentialNodes)

        for sourceNodeId, centralityMeasure in mostInfluentialNodes.items():
            knowledgeImpactFactor = random.choice(knowledgeImpactList)
            
            for neighbor in G.neighbors(sourceNodeId):
                # get node level knowledge about the category
                categoryKnowledge = nx.get_node_attributes(G, categoryValue)
                sourceKnowledge = categoryKnowledge[sourceNodeId]
                destinationKnowledge = categoryKnowledge[neighbor]

                # update the edge weight between the two nodes
                updateEdgeInformation(G, sourceNodeId, neighbor, knowledgeImpactFactor, sourceKnowledge, destinationKnowledge)
                
                # update the knowledge of the neighbor node
                updateNodeKnowledgeForCategory(G, neighbor, knowledgeImpactFactor, sourceKnowledge, destinationKnowledge, categoryValue)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
    return


def updateNodeKnowledgeForCategory(G, neighbor, knowledgeImpactFactor, sourceKnowledge, destinationKnowledge, categoryValue):
    # formula for updating the edge weight is 
    # knowledge = knowledge + (knowledgeImpactFactor) * (absolute change in knowledge)
    G.nodes[neighbor][categoryValue] = max(0, G.nodes[neighbor][categoryValue] + (knowledgeImpactFactor) * (abs(sourceKnowledge - destinationKnowledge)))
    return


def updateEdgeInformation(G, sourceNodeId, neighbor, knowledgeImpactFactor, sourceKnowledge, destinationKnowledge):
    # formula for updating the edge weight is 
    # Wtrust = Wtrust + (knowledgeImpactFactor) * (absolute change in knowledge)
    #G[sourceNodeId][neighbor]["weight"] += (knowledgeImpactFactor) * (abs(sourceKnowledge - destinationKnowledge))
    G[sourceNodeId][neighbor]["weight"] = max(0, G[sourceNodeId][neighbor]["weight"] + (knowledgeImpactFactor) * (abs(sourceKnowledge - destinationKnowledge)))
    return


def deltacon(G1, G2):
    a1 = (nx.adjacency_matrix(G1))
    d1 = diags(np.array(a1.sum(0)).flatten(), dtype=float)
    a2 = (nx.adjacency_matrix(G2))
    d2 = diags(np.array(a2.sum(0)).flatten(), dtype=float)
    # Getting Number of Nodes
    num_nodes = G1.number_of_nodes()
    
    # Creating the Groups
    identity = eye(num_nodes)
    eps = 0.9
    nodes = list(range(0,num_nodes))
    
    # Declaring the number of groups
    num_groups = max(int(num_nodes/100), 10)
    
    # Random Shuffle of nodes
    random.shuffle(nodes)
    groups = [nodes[i::num_groups] for i in range(num_groups)]
    
    # Creating the Affinity Matrix
    s = np.zeros((num_nodes,num_groups))
    for k in range(num_groups):
        for i in groups[k]:
            s[i][k] = 1
    
    # Converting all Matrices to CSR Matrix for computation
    I = csr_matrix(identity)
    
    D1 = csr_matrix(d1)
    A1 = csr_matrix(a1)
    
    A2 = csr_matrix(a2)
    D2 = csr_matrix(d2)
    
    # Calculating the Affinity matrices Equation (Ax = b) Here, b = s, A = I+(D1*pow(eps,2))-(A1*eps)
    S1 = spsolve((I+(D1*pow(eps,2))-(A1*eps)), s)
    S2 = spsolve((I+(D2*pow(eps,2))-(A2*eps)), s)

    # Calculating Root Euclidean distance
    d = np.sqrt(np.sum(np.square(np.sqrt(S1)-np.sqrt(S2))))
    #d = np.sqrt(np.sum(np.sqrt(np.square(S1-S2))))
    # Calculating similarity score
    sim = 1/(1+d)
    return sim



def getMostInfluentialNode(G, categoryValue):
    # get the subgraph of nodes that have the category value > 0
    # find the degree centrality for them
    
    nodeView = list(G.nodes(data = True))
    nodeListByCategory = []
    for node in nodeView:
        # print(node, " -- ", node[1][categoryValue])
        # print("***************************************")
        if(node[1][categoryValue] > 0):
            nodeListByCategory.append(node[0])
    
    # generate the subgraph
    subgraph = G.subgraph(nodeListByCategory)

    # find the centrality for the subgraph
    #centrality = nx.global_reaching_centrality(subgraph, weight='weight')
    centrality = {}
    for node in subgraph.nodes:
        centrality[node] = nx.local_reaching_centrality(subgraph, node)
    #print(centrality)

    # sort the centrality measures
    centrality = sorted(centrality.items(), key=lambda d: d[1], reverse=True)

    # top centrality
    # print(len(centrality))
    max_centrality_measure = max(centrality, key=lambda x:x[1])[1]
    # print(max_centrality_measure)
    max_centrality_dict = {}
    for cent in centrality:
        key = cent[0]
        value = cent[1]
        # print(cent)
        if(value != max_centrality_measure):
            break
        else:
            max_centrality_dict[key] = value

    # if there are multiple nodes with the same centrality, 
    # then choose the node that has the highest knowledge of the category

    # print("length of centrality ", len(max_centrality_dict))
    return max_centrality_dict

if __name__ == "__main__":
    main()
