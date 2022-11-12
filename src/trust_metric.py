import pandas as pd
import os
import networkx as nx
import numpy as np
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

    # iterate through each of the file
    for unprocessedFile in unprocessedFiles:

        dataset = pd.read_csv(basePath + unprocessedFile)
        print(dataset.shape)
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
            
            # store in a dict with graph id (timestamp id) and the value of the metric
            dict = {
                'Graph': G,
                # 'adjacencyMatrix': adjacencyMatrix,
                'graphMetric': graphMetric,
                'time': time,
                # 'edgeList': edgeList,
                "no_of_nodes": len(nodeSet)+1
            }

            graphLevelInformation.append(dict)

        # sort the dict by value to get the top n timestamps
        graphLevelInformation = sorted(graphLevelInformation, key=lambda d: d['graphMetric'], reverse=True) 
        break
    

    # aggregate the graphs in those range groups by performing NORMALIZATION

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
    return graphMetric

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
                destination = "0"
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