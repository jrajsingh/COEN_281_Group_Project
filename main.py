import matplotlib.pyplot as plt
import nil
import pandas as pd
import seaborn as sns
import time

from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def forest_test(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=51)
    start = time.process_time()
    trained_forest = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train)
    print(time.process_time() - start)
    prediction_forest = trained_forest.predict(x_test)
    print(confusion_matrix(y_test, prediction_forest))
    print(classification_report(y_test, prediction_forest))


def read_data_from_directory(wafer_class, machine_step):
    # Create an empty Dataframe to store all data
    df = pd.DataFrame()
    dir_content = listdir("../Wafer_Data/" + wafer_class + "/" + machine_step + "/")
    dir_content.sort()
    for file in dir_content:
        filepath = "../Wafer_Data/" + wafer_class + "/" + machine_step + "/" + file
        print("Reading File {0}".format(file))
        df = df.append(pd.read_csv(filepath))

    df = df.groupby(['WaferID', "STEP ID"]).describe(percentiles=[])
    if 'good' in wafer_class:
        df.insert(loc=len(df.columns), column="CLASS", value='Good')
    elif 'bad' in wafer_class:
        df.insert(loc=len(df.columns), column="CLASS", value='Bad')
    return df


def remove_columns(df):
    # Removing "COUNT" Column
    col_index = len(df.columns) - 1 - 6
    while col_index >= 0:
        print('Removing Column Number: ', col_index)
        print(df.columns[col_index])
        df.drop(df.columns[col_index], axis=1, inplace=True)
        col_index = col_index - 6

    # Removing 50% Columns
    col_index = len(df.columns) - 1 - 2
    while col_index >= 0:
        print('Removing Columns Number: ', col_index)
        #     df.drop(df.columns[col_index], axis=1, inplace=True)
        print(df.columns[col_index])
        col_index = col_index - 5


def run_z1():
    # reading all Z1 Data
    good_z1_df = read_data_from_directory("good_wafer", "Z1_100")
    bad_z1_df = read_data_from_directory("bad_wafer", "Z1_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z1_df.append(bad_z1_df), columns=good_z1_df.columns)

    # Remove Unnecessary Columns
    remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(), annot=True)
    plt.show()

    # Dividing into Inputs and Outputs and run Random Forest Classification
    x_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    x_df = x_df.fillna(0)
    forest_test(x_df, y_df)


def run_z2():
    # reading all Z1 Data
    good_z2_df = read_data_from_directory("good_wafer", "Z2_100")
    bad_z2_df = read_data_from_directory("bad_wafer", "Z2_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z2_df.append(bad_z2_df), columns=good_z2_df.columns)

    # Remove Unnecessary Columns
    remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    plt.figure(figsize=(30, 30))
    sns.heatmap(round(df.corr(), 2), annot=True)
    plt.show()

    # Dividing into Inputs and Outputs and run Random Forest Classification
    x_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    x_df = x_df.fillna(0)
    forest_test(x_df, y_df)


def run_z3():
    # reading all Z1 Data
    good_z2_df = read_data_from_directory("good_wafer", "Z3_100")
    bad_z2_df = read_data_from_directory("bad_wafer", "Z3_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z2_df.append(bad_z2_df), columns=good_z2_df.columns)

    # Remove Unnecessary Columns
    remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    plt.figure(figsize=(30, 30))
    sns.heatmap(round(df.corr(), 2), annot=True)
    plt.show()

    # Dividing into Inputs and Outputs and run Random Forest Classification
    x_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    x_df = x_df.fillna(0)
    forest_test(x_df, y_df)


if __name__ == "__main__":
    run_z1()
    run_z2()
    run_z3()