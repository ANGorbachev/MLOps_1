import pandas as pd

titanic_train = pd.read_csv("./datasets/titanic_train.csv")
titanic_test = pd.read_csv("./datasets/titanic_test.csv")

titanic_train['Age'].fillna(titanic_train['Age'].mean(), inplace=True)
titanic_test['Age'].fillna(titanic_test['Age'].mean(), inplace=True)

pd.DataFrame(titanic_train).to_csv("./datasets/titanic_train.csv", index=False)
pd.DataFrame(titanic_test).to_csv("./datasets/titanic_test.csv", index=False)