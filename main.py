import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Processing the data
parkinsons_data = pd.read_csv('parkinsons.csv')
x = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']

# splitind the data into training & testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2)

# standerlizing the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# training svm model
model = svm.SVC(kernel="linear")
model.fit(x_train, y_train)
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# buulding  Predictive System

input_data = (162.568, 198.346, 77.63, 0.00502, 0.00003, 0.0028, 0.00253, 0.00841, 0.01791, 0.168, 0.00793,
              0.01057, 0.01799, 0.0238, 0.0117, 25.678, 0.427785, 0.723797, -6.635729, 0.209866, 1.957961, 0.135242)
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
    print("The Person does not have Parkinsons Disease")

else:
    print("The Person has Parkinsons")
