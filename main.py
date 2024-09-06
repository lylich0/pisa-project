import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/data.csv')

# Simple linear regression model

# Let's see the relationship between eating the meal with child and child feeling lonely

filtered_df = df.dropna(subset=['ST300Q02JA', 'ST354Q01JA', 'ST352Q08JA', 'ST352Q06JA', 'ST324Q14JA'])

X = filtered_df["ST300Q02JA"].values
y = filtered_df["ST354Q01JA"].values

X = X.reshape(-1, 1)

print(X.shape, y.shape)

plt.scatter(X, y)
plt.xlabel("Frequency of eating meal with child")
plt.ylabel("Frequency of child feeling lonely")
plt.title('Scatter plot')
plt.show()

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
print(predictions[:5])

plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red')

plt.show()

# Result: no linear / weak correlation between the two variables.

# Multiple linear regression model

X = filtered_df[['ST352Q08JA', 'ST352Q06JA', 'ST324Q14JA']].values
y = filtered_df["ST354Q01JA"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:5], y_test[:5]))

r_squared = model.score(X_test, y_test)

rmse = root_mean_squared_error(y_test, y_pred)

print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

kf = KFold(n_splits=6, shuffle=True, random_state=15)

cv_scores = cross_val_score(model, X, y, cv=kf)
print("Cross-validation scores", cv_scores)

print("Mean: ", np.mean(cv_scores))
print("Standard deviation: ", np.std(cv_scores))

# Regularized regression
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

## Ridge: sum of squared coefficients
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)

print("Ridge scores: {}".format(ridge_scores))

## Lasso: sum of absolute values of coefficients
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

lasso_coef = lasso.coef_
print("Lasso coefficient: ", lasso_coef)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))


