import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("train.csv")
# one hot encoding for categorical values
one_hot_encoded_data = pd.get_dummies(data, columns = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18'])

# OPTIONAL
# min-max scaling the data to provide every feature equal weightage
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler_model = scaler.fit(one_hot_encoded_data)
# scaled_data = scaler_model.transform(one_hot_encoded_data)

# distinguishing target and other values
X = one_hot_encoded_data.drop(['target'], axis=1)
y = one_hot_encoded_data['target']
# train validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, random_state=0)
# defining and training model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# printing training and validation accuracy
print(classifier.score(X_train, y_train))
print(classifier.score(X_val, y_val))
