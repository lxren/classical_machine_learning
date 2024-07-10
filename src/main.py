import os
import argparse
import sys
import absl.logging
sys.dont_write_bytecode = True
absl.logging.set_verbosity(absl.logging.ERROR)

from data.build_dataset import build_dataset
from models.build_DecisionTree import build_DecisionTree
from models.build_kNN import build_kNN
from models.build_LR import build_LR

#setup global dataframe variable
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
        model = input("Please select a model between 'DecisionTree','KNearestNeighbors','LogisticRegression':")
        print()
    match model:
        case "DecisionTree":
            build_DecisionTree(df)
        case "KNearestNeighbors":
            n_kfold_str = input("Please provide the number of folds for Stratified K-Fold cross-validator:")
            print()
            n_kfold = int(n_kfold_str)
            build_kNN(df, n_kfold)
        case "LogisticRegression":
            reg = input("Please provide the desired regression / penalty term, 'l1' for Lasso Regression and 'l2' for Ridge Regression:")
            print()
            build_LR(df, reg)
        case _:
            print("Invalid model choice. Please select from 'DecisionTree', 'KNearestNeighbors', or 'LogisticRegression'.")
    
    
    #output = build_model(model)
    #print(output)
