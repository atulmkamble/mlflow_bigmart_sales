# # Big Mart Sales Data

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from matplotlib.pylab import rcParams
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
rcParams['figure.figsize'] = 12, 8
import mlflow
import mlflow.sklearn
import logging
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



if __name__ == "__main__":

        warnings.filterwarnings("ignore")
        np.random.seed(40)

        csv_url_train ="train_modified.csv"
        csv_url_test ="test_modified.csv"

        try:
                        train = pd.read_csv("train_modified.csv")
                        test = pd.read_csv("test_modified.csv")
        except Exception as e:
                        logger.exception("Unable to load training & test CSV, check your path. Error: %s", e)

        target = 'Item_Outlet_Sales'
        IDcol = ['Item_Identifier','Outlet_Identifier']

        n_estimators =int(sys.argv[1]) if len(sys.argv) > 1 else 10
        min_samples_leaf =int(sys.argv[2]) if len(sys.argv) > 3 else 1
        n_jobs =int(sys.argv[3]) if len(sys.argv) > 4 else -1

        with mlflow.start_run():
                ##n_estimators=400
                ##max_depth=6
                ##min_samples_leaf=100
                ##n_jobs=4
                predictors = [x for x in train.columns if x not in [target]+IDcol]
                rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, min_samples_leaf=min_samples_leaf,n_jobs=n_jobs)

                #Fit the algorithm on the data
                rf.fit(train[predictors], train[target])

                #Predict training set:
                train_predictions = rf.predict(train[predictors])

                #Perform cross-validation:
                cv_score = cross_val_score(rf, train[predictors], train[target], cv=20, scoring='neg_mean_squared_error')
                cv_score = np.sqrt(np.abs(cv_score))

                #Print model report:
                mse = metrics.mean_squared_error(train[target].values, train_predictions)
                rmse = np.sqrt(metrics.mean_squared_error(train[target].values, train_predictions))
                mae = mean_absolute_error(train[target].values, train_predictions)
                r2 = r2_score(train[target].values, train_predictions)

                cv_score_mean= np.mean(cv_score)
                cv_score_std=np.std(cv_score)
                cv_score_min=np.min(cv_score)
                cv_score_max=np.max(cv_score)

                #Predict on testing data:
                test[target] = rf.predict(test[predictors])

                coef6 = pd.Series(rf.feature_importances_, predictors).sort_values(ascending=False)

        IDcol.append(target)
        submission = pd.DataFrame({ x: test[x] for x in IDcol})

        print("Random Forest model Parameters")
        print()
        print("n_estimators: %s" % n_estimators)
        print("max_depth: %s" % max_depth)
        print("min_samples_leaf: %s" % min_samples_leaf)
        print("n_jobs: %s" % n_jobs)
        print()
        print("Model Metrics")
        print()
        print("MSE: %s" % mse)
        print("RMSE: %s" % rmse)
        print("MAE: %s" % mae)
        print("R2: %s" % r2)
        print()
        print("Model Cross Validation Metrics")
        print()
        print("cv_score_mean: %s" % cv_score_mean)
        print("cv_score_std: %s" % cv_score_std)
        print("cv_score_min: %s" % cv_score_min)
        print("cv_score_max: %s" % cv_score_max)
        print()

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("n_jobs", n_jobs)


        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("cv_score_mean", cv_score_mean)
        mlflow.log_metric("cv_score_std", cv_score_std)
        mlflow.log_metric("cv_score_min", cv_score_min)
        mlflow.log_metric("cv_score_max", cv_score_max)


        mlflow.log_artifact("Test_Prediction_File.csv", (submission.to_csv("Test_Prediction_File.csv", index=False)))

        features=train.columns
        importances = rf.feature_importances_
        indices = np.argsort(importances)

        plt.figure(1)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance')

        plt.savefig("Feature Importances.png")

        mlflow.log_artifact("Feature Importances.png")

        mlflow.sklearn.log_model(rf, "model")
