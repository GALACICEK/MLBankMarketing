
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_curve, auc


class MLModels:

    def calculate_classifier_metrics(self,clf, clfname:str, x_train,x_test,y_train,y_test):

        y_pred = clf.predict(x_test)
        y_pred_score = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test,y_pred)

        precision = precision_score(y_test,y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        cm = confusion_matrix(y_test,y_pred)


        fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)
        roc_auc = auc(fpr, tpr)

        data = {'classifier_name': [clfname],
                'accuracy': [accuracy],
                'precision': [precision],
                'recall': [recall],
                'f1': [f1],
                'cm': [cm],
                'roc_auc': [roc_auc],
                'fpr': [fpr],
                'tpr': [tpr]}


        return data










