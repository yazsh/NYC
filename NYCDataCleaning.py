import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train = pd.read_csv('/Users/yazen/Desktop/datasets/NYC/train.csv')
test_features = pd.read_csv('/Users/yazen/Desktop/datasets/NYC/test.csv')
print("data loaded")

train_labels = train['trip_duration']

train_features = train.drop('trip_duration',1)
train_features = train_features.drop('pickup_datetime',1)
train_features = train_features.drop('dropoff_datetime',1)
train_features = train_features.drop('id',1)

testids = test_features['id']
test_features = test_features.drop('id',1)
test_features = test_features.drop('pickup_datetime',1)

train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

train_features = pd.get_dummies(train_features, columns=['store_and_fwd_flag', 'vendor_id'])
test_features = pd.get_dummies(test_features, columns=['store_and_fwd_flag', 'vendor_id'])

scalar = preprocessing.MinMaxScaler()
train_features = scalar.fit_transform(train_features)
test_features = scalar.fit_transform(test_features)
train_features, dev_features, train_labels, dev_labels =\
    train_test_split(train_features, train_labels, test_size=.33, random_state=42)
