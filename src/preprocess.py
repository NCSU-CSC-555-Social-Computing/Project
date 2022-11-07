'''
This is a file to be used for one time operation to aggregate all the time data represented as (1/3 rd) second 
into 1 second.
'''
import pandas as pd
import os


def main():

    # Read the data from "datasets/unprocessed"
    basePath = "datasets/unprocessed/"
    unprocessedFiles = getFilesFromFilePath(basePath)

    print("here")

    # iterate through each of the files
    for unprocessedFile in unprocessedFiles:
        
        print(unprocessedFile)
        
        # use pandas to read the csv file as dataFrames
        dataset = pd.read_csv(basePath + unprocessedFile)

        print(dataset.head())
        break


def getFilesFromFilePath(filePath):
    files = os.listdir(filePath)
    return files

if __name__ == "__main__":
    main()