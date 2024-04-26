import os
from sklearn import datasets
import pandas as pd

print("Скачивание набора данных...", end="")
iris = datasets.load_iris()
print("Done!")

X, y, target_names = iris['data'], iris['target'], iris['target_names']

# Создание папки data
if not os.path.exists('data'):
    os.makedirs('data')

print("Сохранение набора данных...", end="")
pd.DataFrame(X).to_csv("./data/train_data.csv", index=False)
pd.DataFrame(y).to_csv("./data/train_target.csv", index=False)
pd.DataFrame(target_names).to_csv("./data/target_names.csv", index=False)
print("Done!")

