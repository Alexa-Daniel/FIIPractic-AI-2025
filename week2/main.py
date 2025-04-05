import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.knn import plot_misclassified_points, plot_classified_points, knn_predict

df=pd.read_csv("data/dataset_hipertensiune2.csv")
X=df[["Varsta", "IMC", "Colesterol"]].values
y=df["Hipertensiune"].values
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
plot_misclassified_points(X_test, y_test, knn_predict(X_train, y_train, X_test, 7))
plot_classified_points(X_train, X_test, y_train, y_test)
print(accuracy_score(y_test, knn_predict(X_train, y_train, X_test, 7)))