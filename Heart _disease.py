import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from serial import Serial
from sklearn import neural_network
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


data = pd.read_csv('heart.csv')
data

{column: len(data[column].unique()) for column in data.columns}

data.info()

numeric_features = ['age', 'sex', 'trtbps',
                    'chol', 'thalachh', 'oldpeak', 'slp', 'caa']

eda_df = data.loc[:, numeric_features].copy()

plt.figure(figsize=(16, 10))

for i in range(len(eda_df.columns)):
    plt.subplot(2, 4, i+1)
    sns.boxplot(eda_df[eda_df.columns[i]])

plt.show()


def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


nominal_features = ['cp', 'slp', 'thall']
dict(zip(['cp', 'slp', 'thall'], ['CP', 'SLP', 'THALL']))


def preprocess_inputs(df, scaler):
    df = df.copy()

    # One hot encoding the nomibnal features
    nominal_features = ['cp', 'slp', 'thall']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SLP', 'THALL'])))

    # Spliting
    y = df['output']
    X = df.drop('output', axis=1).copy()

    # Scale X
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


X, y = preprocess_inputs(data, MinMaxScaler())
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_standard = X_std * (max - min) + min
# X_minmax = (X−Xmin)/(Xmax−Xmin)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0)

# Supervised neural networks


reg = neural_network.MLPClassifier(hidden_layer_sizes=(
    100), activation='relu', solver='adam', alpha=0.00001, learning_rate='adaptive', shuffle=True)
reg.fit(X_train, y_train)

print("Backpropogation neural network accuracy:{:.2f}%".format(
    reg.score(X_test, y_test)*100))

ser = Serial('COM8', 9600, timeout=1)
v1 = int(ser.readline())
time.sleep(0.01)
v2 = int(ser.readline())
slp1 = (v2-v1)/5

v1 = int(ser.readline())
time.sleep(0.01)
v2 = int(ser.readline())
slp2 = (v2-v1)/5

v1 = int(ser.readline())
time.sleep(0.01)
v2 = int(ser.readline())
slp3 = (v2-v1)/5

