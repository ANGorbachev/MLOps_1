import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

X_train = pd.read_csv("./data/train_data.csv")
y_train = pd.read_csv("./data/train_target.csv")

print("Преобразование данных...", end="")
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)

with open('./data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

pd.DataFrame(X_train_transformed).to_csv("./data/train_data_trans.csv", index=False)

print("Done")
