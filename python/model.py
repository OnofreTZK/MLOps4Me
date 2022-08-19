import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

credit = pd.read_csv('Credit.csv')

print(credit.shape)
print(credit.head())

for col in credit.columns:
    if credit[col].dtype == 'object':
        ''' transforming our values from categories to numbers '''
        credit[col] = credit[col].astype('category').cat.codes

print(credit.shape)
print(credit.head())

forecasters = credit.iloc[:,0:20].values

classes = credit.iloc[:,20].values

print(forecasters)

x_training, x_test, y_training, y_test = train_test_split(forecasters, classes,
                                                          test_size=0.3, random_state=123)

mlflow.set_experiment("naive_bayes_experiment")

with mlflow.start_run():
    # Model
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_training, y_training)

    forecasts = naive_bayes.predict(x_test)

    # metrics
    accuracy = accuracy_score(y_test, forecasts)
    recall = recall_score(y_test, forecasts)
    precision = precision_score(y_test, forecasts)
    f1 = f1_score(y_test, forecasts)
    auc = roc_auc_score(y_test, forecasts)
    log = log_loss(y_test, forecasts)

    # register metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("log", log)


    # graphics
    confusion = plot_confusion_matrix(naive_bayes, x_test, y_test)
    plt.savefig("confusion.png")
    roc = plot_roc_curve(naive_bayes, x_test, y_test)
    plt.savefig("roc.png")

    # log graphics
    mlflow.log_artifact("confusion.png")
    mlflow.log_artifact("roc.png")

    # model
    mlflow.sklearn.log_model(naive_bayes, "naive_bayes_model")

    # execution info
    print("Model: ", mlflow.active_run().info.run_uuid)

mlflow.end_run()
