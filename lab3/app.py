import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def classifier(sepal_length, sepal_width, petal_length, petal_width):
    target_names = np.array(pd.read_csv("./data/target_names.csv")).reshape(3)

    with open('./data/scaler.pkl', 'rb') as f1:
        scaler = pickle.load(f1)

    with open('./model/model.pkl', 'rb') as f2:
        model = pickle.load(f2)

    X_test = np.array([sepal_length, sepal_width, petal_length, petal_width])
    X_test = scaler.transform((X_test, ))
    y_pred = model.predict(X_test)
    return target_names[y_pred[0]]

st.title('Iris classification')

# Ввод данных
with st.form('Iris classificator'):
    sepal_length = st.number_input('Insert a sepal length (cm)', value=5.1)
    sepal_width = st.number_input('Insert a sepal width (cm)', value=3.5)
    petal_length = st.number_input('Insert a petal length (cm)', value=1.4)
    petal_width = st.number_input('Insert a petal width (cm)', value=0.2)
    submit = st.form_submit_button('Классифицировать!')

if submit:
    iris_class = classifier(sepal_length, sepal_width, petal_length, petal_width)
    st.subheader(f'Classification result: {iris_class}')

st.text('Examples of irises:\n'
        'setosa = 5.1, 3.3, 1.7, 0.5\n'
        'versicolor = 6.1, 2.8, 4.7, 1.2\n'
        'virginica = 6.3, 2.7, 4.9, 1.8\n')
