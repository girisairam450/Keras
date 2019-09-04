import pandas as pd
from keras.models import load_model

model = load_model('trained_model.h5')

X = pd.read_csv('proposed_new_product.csv').values
prediction = model.predict(X)
prediction = prediction[0][0]

prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print('Earnings Prediction for proposed product - ${}'.format(prediction))
