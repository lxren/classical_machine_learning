import os
import argparse
import sys
sys.dont_write_bytecode = True
from data.build_dataset import build_dataset
from models.build_DecisionTree import build_DecisionTree

df = build_dataset()

# build_DecisionTree(df)

if __name__ == '__main__':

    # build argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', help=f"Select a model between 'DecisionTree','KNearestNeighbors','LogisticRegression'")
    args = parser.parse_args()
    model = args.model

    # verbose to print df head
    if args.verbose:
        print(df.head())

    # accept arg to select and run ML model
    if not model:
        model = input("Please select a model between 'DecisionTree','KNearestNeighbors','LogisticRegression'")
    match model:
        case "DecisionTree":
            build_DecisionTree(df)
        case "KNearestNeighbors":
            print('KNN')
        case "LogisticRegression":
            print("LR")
    
    
    #output = build_model(model)
    #print(output)
