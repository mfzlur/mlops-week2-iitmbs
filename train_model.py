import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)  
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
    
    joblib.dump(clf, model_path)

if __name__ == '__main__':
    import sys
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    train_model(data_path, model_path)
