from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def build_LR(df,reg):

    y = df['target']
    X = df.iloc[:,1:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear', penalty = reg)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)

    print(f"Classification report for -{reg}- regularization:/n{classification_report(y_test, y_pred)}")

if __name__ == '__main__':
    build_LR()