import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

X_train = pd.read_csv("./data/train_data_trans.csv")
y_train = np.array(pd.read_csv("./data/train_target.csv")).reshape(X_train.shape[0])

print("Обучение и сохранение модели машинного обучения...", end='')
classifier = RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
classifier.fit(X_train, y_train)

try:
    if not os.path.exists('model'):
        os.makedirs('model')

    with open('./model/model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    print("Done")
except:
    print('Ошибка!')


