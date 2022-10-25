import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('iris.data')
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

#making the catagorical value into numerical
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

#spliting the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#building the model
sv = SVC(kernel='linear').fit(X_train,y_train)

#pickle file dump
pickle.dump(sv, open('iri.pkl', 'wb'))
