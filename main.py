import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import root_mean_squared_error, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve

df = pd.read_csv('data/data.csv')

# Simple linear regression model

# Let's see the relationship between eating the meal with child and child feeling lonely

filtered_df = df.dropna(subset=['ST300Q02JA', 'ST354Q01JA', 'ST352Q08JA', 'ST352Q06JA', 'ST324Q14JA'])

# Creating features

## Before making any predictions, I need to create the feature and target arrays

X = filtered_df["ST300Q02JA"].values
y = filtered_df["ST354Q01JA"].values

X = X.reshape(-1, 1)

print(X.shape, y.shape)

# Building a linear regression model

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

# Visualizing a linear regression model

plt.scatter(X, y, color='blue')

# Red line plot displaying the predictions against X

plt.plot(X, predictions, color='red')

plt.xlabel("Eating together")
plt.ylabel("Feeling lonely")
plt.show()

# Result:

# Multiple linear regression model

X = filtered_df[['ST352Q08JA', 'ST352Q06JA', 'ST324Q14JA']].values
y = filtered_df["ST354Q01JA"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:5], y_test[:5]))

# Regression performance
r_squared = model.score(X_test, y_test)
print("R^2: {}".format(r_squared))

rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE: {}".format(rmse))

# Cross-validation for R-squared
kf = KFold(n_splits=6, shuffle=True, random_state=15)

cv_scores = cross_val_score(model, X, y, cv=kf)
print("Cross-validation scores", cv_scores)

# Analyzing cross-validation metrics
print("Mean: ", np.mean(cv_scores))
print("Standard deviation: ", np.std(cv_scores))
print(np.quantile(cv_scores, [0.025, 0.975]))

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

# Create a DataFrame with the feature data and add a column for feature names
df_features = ['ST352Q08JA', 'ST352Q06JA', 'ST324Q14JA']

plt.bar(df_features, lasso_coef)
plt.xticks(rotation=45)
plt.show()

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))

# Building a logistic regression model

threshold = 3

X_train_binary = (X_train >= threshold).astype(int)
X_test_binary = (X_test >= threshold).astype(int)
y_train_binary = (y_train >= threshold).astype(int)
y_test_binary = (y_test >= threshold).astype(int)

logreg = LogisticRegression()
logreg.fit(X_train_binary, y_train_binary)

y_pred_probs = logreg.predict_proba(X_test_binary)[:, 1]
print(y_pred_probs[:5])

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve for loneliness regression")
plt.show()

# ROC AUC

print("ROC AUC score: ", roc_auc_score(y_test_binary, y_pred_probs))
print("Confusion matrix: ", confusion_matrix(y_test_binary, y_pred))
print("Classification report: ", classification_report(y_test_binary, y_pred))

# Optimizing model: hyperparameter tuning

param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

## Grid Search

lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)
lasso_cv.fit(X_train, y_train)

print("Tuned lasso parameters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))

## Randomized Search

params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}

logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)
logreg_cv.fit(X_train, y_train)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))

