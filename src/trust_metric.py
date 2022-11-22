import pandas as pd
import os
import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, eye
import matplotlib.pyplot as plt
import csv
'''
this file traverses through the processed dataset and takes a weighted sum of edges and components
then find the top n time values
if there are group of values present, normalize the edge weights and assign values
show in networkx with the finalized version of graph
'''
def main():
    # read through all the processed data

    basePath = "../datasets/processed/"
    unprocessedFiles = getFilesFromFilePath(basePath)

    # adjacency matrix
    
    # Initialize an N*N matrix
    row_count = 0
    col = 100
    networkTimeStampMatrix = [[0 for i in range(col)] for j in range(len(unprocessedFiles)+1)]
    col_list = getColumnList(col)

    node_count = 0
    masterGraphEdgeList = []
    # iterate through each of the file
    for unprocessedFile in unprocessedFiles:

        dataset = pd.read_csv(basePath + unprocessedFile)
        print(unprocessedFile)
        # for a single file, iterate through each row (each row will be a graph here)
        
        graphLevelInformation = []
        for index, row in dataset.iterrows():
            edgeList, time, nodeSet = generateEdgeList(row)

            adjacencyMatrix = createAdjacencyMatrix(edgeList, nodeSet)

            # generate graph for every edgelist
            G = nx.DiGraph()
            G.add_weighted_edges_from(edgeList)

            # for every node, get the sum of weight of all the edges from the node and multiply it with the degree of node
            # get the overall sum of that which would be the final graph

            graphMetric = computeGraphMetric(adjacencyMatrix)
            # if(time < 3):
            #     visualizeNetworkXGraph(G, "network_" + str(row_count) + "_time_" + str(time) + ".png")
            
            # store in a dict with graph id (timestamp id) and the value of the metric
            dict = {
                'Graph': G,
                # 'adjacencyMatrix': adjacencyMatrix,
                'graphMetric': graphMetric,
                'time': time,
                'edgeList': edgeList,
                "no_of_nodes": len(nodeSet)
            }

            graphLevelInformation.append(dict)

        # sort the dict by value to get the top n timestamps
        top5TimeStampsGraph = sorted(graphLevelInformation, key=lambda d: d['graphMetric'], reverse=True)[:5] 

        # sort it by time
        top5TimeStampsGraph = sorted(top5TimeStampsGraph, key=lambda d: d['time'], reverse = True)

        maximumEngagementGraphList = getMaximumEngagementGraphList(top5TimeStampsGraph)

        # iterate through the graph list and take the average of edge weights to construct one graph
        finalEngagementEdgeList, no_of_nodes = getFinalEngagementGraph(maximumEngagementGraphList)
        # print(no_of_nodes)
        
        for edge in finalEngagementEdgeList:
            sourceNode = edge[0] + node_count
            destinationNode = edge[1] + node_count
            masterGraphEdgeList.append((sourceNode, destinationNode, edge[2]))
        


        
        # for i in range(0,5):
        #     info = graphLevelInformation[i]
        #     # update the matrix
        #     col = int(info["time"])
        #     networkTimeStampMatrix[row_count][col] = 1

        row_count += 1
        node_count += no_of_nodes
    
    writeMasterEdgeListToCSV('../datasets/interpolation/masterEdgeList.csv', masterGraphEdgeList)
    # print(masterGraphEdgeList)

    # df = pd.DataFrame(networkTimeStampMatrix, columns = col_list)
    # df.to_csv("../datasets/interpolation/networkTimestamp.csv")

    # aggregate the graphs in those range groups by performing NORMALIZATION


def writeMasterEdgeListToCSV(fileName, masterGraphEdgeList):
    with open(fileName,'w') as out:
        csv_out=csv.writer(out)
        # csv_out.writerow(['name','num'])
        for edge in masterGraphEdgeList:
            csv_out.writerow(edge)

def getFinalEngagementGraph(maximumEngagementGraphList):
    edges = []
    finalEdgeList = []
    no_of_nodes = 0
    for graph in maximumEngagementGraphList:
        edgeList = graph['edgeList']
        # print(edgeList)
        edges.append(edgeList)
        no_of_nodes = graph['no_of_nodes']
    
    # print(len(edges[0]))
    for i in range (len(edges[0])):
        wt = 0
        source = -1
        destination = -1

        for j in range(len(edges)):
            source = edges[j][i][0]
            destination = edges[j][i][1]
            wt += edges[j][i][2]
        wt/= len(edges)
        wt = round(wt, 2)

        finalEdgeList.append((source, destination, wt))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(finalEdgeList)
    return finalEdgeList, no_of_nodes

def getMaximumEngagementGraphList(top5TimeStampsGraph):
    # if there are timestamps within range of 3 secs, then we combine those graphs
    maxi = -1
    vals = []
    count = 1
    vals.append(top5TimeStampsGraph[0])
    final = []
    for i in range(len(top5TimeStampsGraph)-1):
        graphI = top5TimeStampsGraph[i]
        graphI1 = top5TimeStampsGraph[i+1]
        # print(graphI['time'], "---", graphI1['time'])
        if(graphI['time'] - graphI1['time'] <=2):
            count +=1
            vals.append(graphI1)
        else:
            if (count > maxi):
                final = []
                final.extend(vals)
                maxi = count
            vals = []
            vals.append(graphI1)            
            count = 1
    
    if(count > maxi):
        final = []
        final.extend(vals)
    # print(final)
    # print("************************************")
    return final

def visualizeNetworkXGraph(G, fileName):
    pos = nx.spring_layout(G)
    # print(G.nodes())
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    weights = tuple(10*x for x in weights)
    # print("*************")
    nx.draw(G, pos, with_labels=True, edgelist = G.edges(), node_color='orange', edge_color=weights, width = weights, edge_cmap=plt.cm.Blues)
    plt.savefig("../assets/plots/" + fileName)
    plt.clf()
    return

def getColumnList(col):
    col_list = []
    for i in range(0, col):
        col_list.append(i)
    return col_list

def computeGraphMetric(adjacencyMatrix):
    graphMetric = 0
    for i in range (len(adjacencyMatrix)):
        deg = 0
        edgeWeightSum = 0
        for j in range (len(adjacencyMatrix[0])):
            if(adjacencyMatrix[i][j] != 0):
                deg+=1
                edgeWeightSum += adjacencyMatrix[i][j]
        graphMetric += (deg * edgeWeightSum)
    return round(graphMetric, 2)

def createAdjacencyMatrix(edgeList, nodeSet):
    rows = cols = len(nodeSet)+1
    arr = [[0 for i in range(cols)] for j in range(rows)]

    for edge in edgeList:
        x = edge[0]
        y = edge[1]
        weight = edge[2]
        arr[x][y] = weight
    return arr

def generateEdgeList(rowInfo):
    time = 0
    edgeList = []
    nodeSet = set()
    for columnIndex, value in rowInfo.items():
        edge_name = columnIndex.split("_")
        if(edge_name[0] == "Unnamed: 0"):
            continue
        elif(edge_name[0] == "TIME"):
            time = value
        else:
            source = edge_name[0].replace("P", "")
            if(edge_name[2] == "LAPTOP"):
                continue
                # destination = "0"
            else:
                destination = edge_name[2].replace("P", "")
            
            if source == destination:
                continue
            nodeSet.add(int(source))
            edgeList.append((int(source), int(destination), float(value)))
    return edgeList, time, nodeSet

'''
Function to get files from path
'''
def getFilesFromFilePath(filePath):
    files = os.listdir(filePath)
    return files


if __name__ == "__main__":
    main()