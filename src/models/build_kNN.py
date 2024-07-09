from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import DistanceMetric
from sklearn.metrics import accuracy_score
import pandas as pd

def build_kNN(df, n_kfold):

    y = df['target']
    X = df.iloc[:,1:-1]
    scaler = StandardScaler()
    scaler.fit(X)

    neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    distances = ['euclidean','manhattan','chebyshev']
    results = {}

    for n in neighbors:
        for d in distances:
            skf = StratifiedKFold(n_splits=n_kfold)
            knn = KNeighborsClassifier(n_neighbors = n, metric = d)

            labels = []
            predictions = []
            accuracy_results = []

            for i, (train_index, test_index) in enumerate(skf.split(X,y)):
                fold_X_train = X.iloc[train_index,:]
                fold_y_train = y.iloc[train_index]
                fold_X_test = X.iloc[test_index,:]
                fold_y_test = y.iloc[test_index]

                model = KNeighborsClassifier()
                model.fit(fold_X_train,fold_y_train)

                predictions.extend(model.predict(fold_X_test))

                labels.extend(fold_y_test)
                
                accuracy = accuracy_score(labels,predictions) #using accuracy instead of precision as question dictates; if precision then just use classification_report
                accuracy_results.append(accuracy)
            
            results[f'neighbour-{n}-distance-{d}'] = accuracy_results
    results_df = pd.DataFrame.from_dict(results)
    print(results_df.mean(axis=0).sort_values(ascending = False))

if __name__ == '__main__':
    build_kNN()