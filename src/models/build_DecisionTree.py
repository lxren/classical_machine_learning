from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import os

def build_DecisionTree(df):
    y = df['target']
    X = df.iloc[:,1:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f'accurancy: {metrics.accuracy_score(y_test,y_pred):.2f} recall: {metrics.recall_score(y_test,y_pred):.2f} precision: {metrics.precision_score(y_test,y_pred):.2f}')
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    heatmap = sns.heatmap(confusion_matrix,annot=True)
    plt.show()
    plt.title('Decision Tree Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('./src/visualization/DecisionTree_ConfusionMatrix.png')

if __name__ == '__main__':
    build_DecisionTree()