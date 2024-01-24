# MLBANKMARKETING

This project of purpose is using bank marketing datasets and build machine learning models and than calculate evaluation metrics and dessicion which model functional.


- [Data Source](http://archive.ics.uci.edu/dataset/222/bank+marketing)

- Look to understand datasets :
    - [First information](https://github.com/GALACICEK/MLBankMarketing/blob/main/datas/bank-additional-names.txt)
    - [Secondary information](https://github.com/GALACICEK/MLBankMarketing/blob/main/datas/bank-names.txt)

 ### Requirements :

Generate virtual env on your project. Also you could need that libs, check your libs.

    - pip install pandas
    - pip install numpy
    - pip install matplotlib
    - pip install scikit-learn
    - pip install xgboost 

Also can write just this code on your terminal:

    - pip install -r requirements.txt



---

### MLBANKMARKETING Project Files
 - [main.ipynb](https://github.com/GALACICEK/MLBankMarketing/blob/main/main.ipynb)
 - [data_collection_and_cleaning.py](https://github.com/GALACICEK/MLBankMarketing/blob/main/data_collection_and_cleaning.py)
 - [exploratory_data_analysis.py](https://github.com/GALACICEK/MLBankMarketing/blob/main/exploratory_data_analysis.py)
 - [machine_learning_models.py](https://github.com/GALACICEK/MLBankMarketing/blob/main/machine_learning_models.py)
 - [datasets](https://github.com/GALACICEK/MLBankMarketing/tree/main/datas)

### Models

- Using Classification Models:
    - Decision Tree
    - Random Forests
    - KNN
    - SVM
    - Logistic Regression
    - GradientBoostingClassifier
    - XGBClassifier
    - GaussianNB 

## Summary

- Info DataSets :
    ---
         <class 'pandas.core.frame.DataFrame'>
         RangeIndex: 41188 entries, 0 to 41187
          Data columns (total 21 columns):

        |    |Column          |Non-Null Count  |Dtype   |
        |----| ---            | ---            | ---    |
        | 0  | age            | 41188 non-null | int64  |
        | 1  | job            | 41188 non-null | object |
        | 2  | marital        | 41188 non-null | object |
        | 3  | education      | 41188 non-null | object |
        | 4  | default        | 41188 non-null | object |
        | 5  | housing        | 41188 non-null | object |
        | 6  | loan           | 41188 non-null | object |
        | 7  | contact        | 41188 non-null | object |
        | 8  | month          | 41188 non-null | object |
        | 9  | day_of_week    | 41188 non-null | object |
        | 10 | duration       | 41188 non-null | int64  |
        | 11 | campaign       | 41188 non-null | int64  |
        | 12 | pdays          | 41188 non-null | int64  |
        | 13 | previous       | 41188 non-null | int64  |
        | 14 | poutcome       | 41188 non-null | object |
        | 15 | emp.var.rate   | 41188 non-null | float64|
        | 16 | cons.price.idx | 41188 non-null | float64|
        | 17 | cons.conf.idx  | 41188 non-null | float64|
        | 18 | euribor3m      | 41188 non-null | float64|
        | 19 | nr.employed    | 41188 non-null | float64|
        | 20 | deposit        | 41188 non-null | object |

        dtypes: float64(5), int64(5), object(11)
        memory usage: 6.6+ MB

- Deleted Duplicates: 12
- Clean Data for unknown label
- EDA , Exploratory Data Analysis
![categorical Count Plot](./Outputs/countplot_categorical_output.png)

![categorical_relationship_output](./Outputs/categorical_relationship_output.png)

![age_distrubution_output](./Outputs/age_distrubution_output.png)

![corr_heatmap_output](./Outputs/corr_heatmap_output.png)

- LabelEncoding

- Using train,test columns : 
    - ['age', 'job', 'marital', 'education','housing', 'loan', 'month', 'day_of_week', 'duration',
      
- Reports:

    - DT_Bagging 
    accuracy value :  0.909  precision :  0.619  recall:  0.523  f1_score :  0.567 
    -----
    - Random_Forest 
    accuracy value :  0.910  precision :  0.638  recall:  0.493  f1_score :  0.556 
    -----
    - KNeighbors 
    accuracy value :  0.904  precision :  0.668  recall:  0.304  f1_score :  0.418 
    -----
    - SVC-rbf 
    accuracy value :  0.909  precision :  0.674  recall:  0.388  f1_score :  0.493 
    -----
    - SVC-sigmoid 
    accuracy value :  0.861  precision :  0.388  recall:  0.383  f1_score :  0.386 
    -----
    - Logistic_Regression 
    accuracy value :  0.906  precision :  0.648  recall:  0.378  f1_score :  0.477 
    -----
    - Gradient_Boosting 
    accuracy value :  0.913  precision :  0.658  recall:  0.486  f1_score :  0.559 
    -----
    - XGBC 
    accuracy value :  0.910  precision :  0.626  recall:  0.527  f1_score :  0.572 
    -----
    - GaussianNB 
    accuracy value :  0.869  precision :  0.444  recall:  0.586  f1_score :  0.505 
    -----

![cm_output](./Outputs/cm_output.png)

![auc-roc_graph_output](./Outputs/auc-roc_graph_output.png)

- It should not be selected based on GaussianNB and SVC-sigmoid auc values. Other models available.
- XGBC, Gradient_Boosting, Random_Forest and DT_Bagging models can be used by looking at f1 scores.
- Since the split operation is random, different results are obtained each time.
