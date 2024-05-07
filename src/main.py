from data.build_dataset import build_dataset
from models.build_DecisionTree import build_DecisionTree
import sys
sys.dont_write_bytecode = True

df = build_dataset()

print(df.head())

build_DecisionTree(df)

