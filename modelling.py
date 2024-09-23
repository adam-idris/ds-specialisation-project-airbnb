from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df = pd.read_csv('cleaned_tabular_data.csv')
df = df.drop(['Unnamed: 0'], axis=1)

features, labels = load_airbnb(df, 'Price_Night')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

sgd_regressor = SGDRegressor(random_state=42)
sgd_regressor.fit(X_train, y_train)

sgd_predictions = sgd_regressor.predict(X_test)

sgd_train_mse = mean_squared_error(y_train, sgd_predictions)
sgd_train_r2 = r2_score(y_train, sgd_predictions)

sgd_mse = mean_squared_error(y_test, sgd_predictions)
sgd_r2 = r2_score(y_test, sgd_predictions)

print('Train RMSE:',sgd_train_mse, 'Train R2:', sgd_train_r2)
print('RMSE:',sgd_mse, 'R2:', sgd_r2)