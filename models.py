import keras
import numpy as np
import pickle
import sys
from sklearn.metrics import accuracy_score

sys.path.append('./')
import data_info

def data_(data):
    return data['colors'], data['classes'], data['img'], data['names']

def classify(classification):
    if classification == 'flevo_l':
        model = keras.models.load_model('../models/flevo_l_model')
        with open('../flevo_l/flevo_l_6_7_patchdata.pkl', 'rb') as f:
            data = pickle.load(f)
        y = model.predict(data)
        y = np.argmax(y, axis = 1)
        colors ,classes, img, unique = data_(data_info.flevo_l)

        with open('../flevo_l/flevo_l_6_7.pkl', 'rb') as f:
            x_train, y_train,  x_test, y_test = pickle.load(f)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        accuracy = (accuracy_score(y_pred, y_test))

    elif classification == 'flevo_c':
        model = keras.models.load_model('../models/flevo_c_model')
        with open('../flevo_c/flevo_c_6_21_patchdata.pkl', 'rb') as f:
            data = pickle.load(f)
        y = model.predict(data)
        y = np.argmax(y, axis = 1)
        colors ,classes, img, unique = data_(data_info.flevo_c)

        with open('../flevo_c/flevo_c_6_21.pkl', 'rb') as f:
            x_train, y_train,  x_test, y_test = pickle.load(f)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        accuracy = (accuracy_score(y_pred, y_test))

    elif classification == 'sfbay_l':
        model = keras.models.load_model('../models/sfbay_l_model')
        with open('../sfbay_l/sfbay_l_4_21_patchdata.pkl', 'rb') as f:
            data = pickle.load(f)
        y = model.predict(data)
        y = np.argmax(y, axis = 1)
        colors ,classes, img, unique = data_(data_info.sfbay_l)

        with open('../sfbay_l/sfbay_l_4_21.pkl', 'rb') as f:
            x_train, y_train,  x_test, y_test = pickle.load(f)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        accuracy = (accuracy_score(y_pred, y_test))

    elif classification == 'sfbay_c':
        model = keras.models.load_model('../models/sfbay_c_model')
        with open('../sfbay_c/sfbay_c_4_21_patchdata.pkl', 'rb') as f:
            data = pickle.load(f)
        y = model.predict(data)
        y = np.argmax(y, axis = 1)
        colors ,classes, img, unique = data_(data_info.sfbay_c)

        with open('../sfbay_c/sfbay_c_4_21.pkl', 'rb') as f:
            x_train, y_train,  x_test, y_test = pickle.load(f)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        accuracy = (accuracy_score(y_pred, y_test))  

    elif classification == 'general':
        model = keras.models.load_model('../models/generalize_model')
        with open('../sfbay_l/sfbay_l_4_21_patchdata.pkl', 'rb') as f:
            data = pickle.load(f)
        y = model.predict(data)
        y = np.argmax(y, axis = 1)
        colors ,classes, img, unique = data_(data_info.sfbay_l)

        with open('../flevo_l/flevo_l_6_7.pkl', 'rb') as f:
            x_train, y_train,  x_test, y_test = pickle.load(f)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        accuracy = (accuracy_score(y_pred, y_test))

    data = {"colors": colors, "classes": classes, "y": y, "img": img}
    return data, unique, accuracy