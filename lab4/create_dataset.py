import pandas as pd
from catboost.datasets import titanic
titanic_train, titanic_test = titanic()

titanic_train = titanic_train[['PassengerId', 'Survived', 'Name', 'Pclass', 'Sex', 'Age']]
titanic_test = titanic_test[['PassengerId', 'Name', 'Pclass', 'Sex', 'Age']]

pd.DataFrame(titanic_train).to_csv("./datasets/titanic_train.csv", index=False)
pd.DataFrame(titanic_test).to_csv("./datasets/tdvcitanic_test.csv", index=False)
