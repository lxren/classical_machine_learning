### Install libraries
```bash
pip install .
```
### Download data for local copy (optional)
```bash
python src/data/download_dataset.py
```
### Build dataset (build directly from json file using mlcroissant)
```bash
python src/data/build_dataset.py
```
### Print & preview first 5 rows of dataset
```bash
python ./src/main.py --verbose
```
### Build Model (choose between DecisionTree, KNearestNeighbors, and LogisticRegression)
```bash
python ./src/main.py --model <model_name>

model_name: DecisionTree|KNearestNeighbors|LogisticRegression
```