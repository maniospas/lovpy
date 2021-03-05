from logipy.wrappers import LogipyPrimitive, logipy_call
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = logipy_call(read_csv,url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = logipy_call(train_test_split,X, y, test_size=0.20, random_state=1)

# Make predictions on validation dataset
model = logipy_call(SVC,gamma='auto')
logipy_call(model.fit,X_train, Y_train)
predictions = logipy_call(model.predict,X_validation)
# Evaluate predictions
logipy_call(print,logipy_call(accuracy_score,Y_validation, predictions))
logipy_call(print,logipy_call(confusion_matrix,Y_validation, predictions))
logipy_call(print,logipy_call(classification_report,Y_validation, predictions))