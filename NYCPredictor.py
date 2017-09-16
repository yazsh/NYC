from NYCDataCleaning import *
import keras as ks


model = ks.models.Sequential()
model.add(ks.layers.Dense(32,activation='relu', input_shape=(9,)))
model.add(ks.layers.Dense(32,activation='relu'))
model.add(ks.layers.Dense(32,activation='relu'))
model.add(ks.layers.Dense(1,activation='relu'))


model.compile(optimizer=ks.optimizers.sgd(), loss='mean_squared_logarithmic_error')
model.fit(train_features, train_labels, batch_size=1000, epochs=20 )
model.evaluate(dev_features, dev_labels)
data = pd.DataFrame(model.predict(test_features))

data.columns = ["trip_duration"]

data = pd.concat([testids,data],axis=1)

data.to_csv("/Users/yazen/Desktop/datasets/NYC/prediction.csv", index=False)
