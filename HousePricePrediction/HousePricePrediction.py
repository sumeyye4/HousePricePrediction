import pandas as pd
from sklearn.linear_model import LinearRegression


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.columns

model = LinearRegression()

x_train = train.drop('SalePrice', axis=1)
y_train = train.loc[:, 'SalePrice']

model.fit(x_train, y_train)

x_test = test.drop('SalePrice', axis=1)
y_test = test.loc[:, 'SalePrice']

predictions = model.predict(x_test)

comparison = pd.DataFrame({"Actual Values": y_test, "Predictions": predictions})

comparison.head()

comparison.tail()