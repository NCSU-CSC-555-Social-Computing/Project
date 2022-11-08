'''
This is a file to be used for one time operation to aggregate all the time data represented as (1/3 rd) second 
into 1 second.
'''
import pandas as pd
import os


def main():

    # Read the data from "datasets/unprocessed"
    basePath = "../datasets/unprocessed/"
    unprocessedFiles = getFilesFromFilePath(basePath)

    for unprocessedFile in unprocessedFiles:
        
        # use pandas to read the csv file as dataFrames
        dataset = pd.read_csv(basePath + unprocessedFile)
        
        i = 0
        new_dataset = []
        while(i<len(dataset)):
            new_row = []
            for col in dataset.columns:
                avg = round((dataset[col][i]+dataset[col][i+1]+dataset[col][i+2])/3,2)
                new_row.append(avg)
            new_dataset.append(new_row)
            new_row[0] = int(i/3)
            i += 3
        new_dataset = (pd.DataFrame(new_dataset))
        new_dataset.columns = dataset.columns
        new_dataset.to_csv('../datasets/processed/processed_'+unprocessedFile)


def getFilesFromFilePath(filePath):
    files = os.listdir(filePath)
    return files

if __name__ == "__main__":
    main()