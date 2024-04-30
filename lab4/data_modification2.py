import pandas as pd
from sklearn.preprocessing import OneHotEncoder

titanic_train = pd.read_csv("./datasets/titanic_train.csv")
titanic_test = pd.read_csv("./datasets/titanic_test.csv")

encoder = OneHotEncoder()
enc_df = pd.DataFrame(encoder.fit_transform(titanic_train[['Sex']]).toarray())
titanic_train = pd.concat([titanic_train, enc_df], axis=1)

pd.DataFrame(titanic_train).to_csv("./datasets/titanic_train.csv", index=False)
pd.DataFrame(titanic_test).to_csv("./datasets/titanic_test.csv", index=False)