import matplotlib.pyplot as plt
import numpy as np
import nil
import pandas as pd
import seaborn as sns
import time

from os import listdir
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def forest_test(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=101)
    start = time.process_time()
    trained_forest = RandomForestClassifier(n_estimators=700).fit(x_train, y_train)
    print(time.process_time() - start)
    prediction_forest = trained_forest.predict(x_test)
    print(confusion_matrix(y_test, prediction_forest))
    print(classification_report(y_test, prediction_forest))


def read_data_from_directory(wafer_class, machine_step):
    # Create an empty Dataframe to store all data
    df = pd.DataFrame()
    dir_content = listdir("../Wafer_Data/" + wafer_class + "/" + machine_step + "/")
    dir_content.sort()
    cnt = 0
    for file in dir_content:
        filepath = "../Wafer_Data/" + wafer_class + "/" + machine_step + "/" + file
        print("Reading File {0}".format(file))
        df = df.append(pd.read_csv(filepath))

    wafer_class_bool = nil
    if 'good' in wafer_class:
        df.insert(loc=len(df.columns), column="CLASS", value='Good')
    elif 'bad' in wafer_class:
        df.insert(loc=len(df.columns), column="CLASS", value='Bad')
    return df


# reading all Z1 Data
good_z1_df = read_data_from_directory("good_wafer", "Z1_100")
bad_z1_df = read_data_from_directory("bad_wafer", "Z1_100")

ndf = good_z1_df.drop(['TIME', 'STEP ID', 'VEL', 'ACC', 'CLASS'], axis=1)
plt.figure(figsize=(20, 20))
sns.heatmap(ndf.corr(), annot=True)
plt.show()

df = pd.DataFrame(good_z1_df.append(bad_z1_df), columns=good_z1_df.columns)

x = df.dropna().reset_index().iloc[:, :-5]
y = df.dropna().reset_index().iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)
trained_forest = RandomForestClassifier(n_estimators=300).fit(x_train, y_train)

prediction_forest = trained_forest.predict(x_test)
print(confusion_matrix(y_test, prediction_forest))
print(classification_report(y_test, prediction_forest))