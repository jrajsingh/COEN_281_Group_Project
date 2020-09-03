import matplotlib.pyplot as plt
import nil
import pandas as pd
import seaborn as sns
import time

from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def forest_test(df):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X = X.fillna(0)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=51)
    start = time.process_time()
    trained_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(x_train, y_train)
    print("Time taken to train model : ", time.process_time() - start)
    prediction_forest = trained_forest.predict(x_test)
    print(confusion_matrix(y_test, prediction_forest))
    print(classification_report(y_test, prediction_forest))

    feature_imp = pd.Series(trained_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_imp)

    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels in your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.figure(figsize=(100, 100))
    plt.show()
    return prediction_forest


def read_data_from_directory(wafer_class, machine_step):
    # Create an empty Dataframe to store all data
    df = pd.DataFrame()
    dir_content = listdir("../Wafer_Data/" + wafer_class + "/" + machine_step + "/")
    dir_content.sort()
    cnt = 0
    print("Start Reading Files")
    for file in dir_content:
        filepath = "../Wafer_Data/" + wafer_class + "/" + machine_step + "/" + file
        #         print("Reading File {0}".format(file))
        df = df.append(pd.read_csv(filepath))

    print("Finished Reading Files")
    df = df.groupby(['WaferID', "STEP ID"]).describe(percentiles=[])
    wafer_class_bool = nil
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
        df.drop(df.columns[col_index], axis=1, inplace=True)
        col_index = col_index - 6

    # Removing 50% Columns
    col_index = len(df.columns) - 1 - 2
    while col_index >= 0:
        print('Removing Columns Number: ', col_index)
        df.drop(df.columns[col_index], axis=1, inplace=True)
        print(df.columns[col_index])
        col_index = col_index - 5

    return df


def find_threshold(results, threshold):
    result_list = []
    current = results[0]
    count = 0
    for value in results:
        if value == current:
            count += 1
        else:
            result_list.append((current, count))
            current = value
            count = 1
    result_list.append((current, count))

    consecutive_found = False
    for key, val in result_list:
        if key == 'Bad' and val > threshold:
            consecutive_found = True
            if val > count:
                count = val
    #             break
    if consecutive_found:
        print("There are about", count,
              "bad wafers all consecutively bad!\n\tPreventative Maintenance is suggested to prevent further losses.")
    else:
        print("The threshold of ", threshold, " consecutively bad wafers were not found.")
    return result_list


def print_heatmap(corr_matrix):
    plt.figure(figsize=(30, 30))
    sns.heatmap(round(corr_matrix, 2), annot=True)
    plt.show()


def run_z1(threshold):
    # reading all Z1 Data
    good_z1_df = read_data_from_directory("good_wafer", "Z1_100")
    bad_z1_df = read_data_from_directory("bad_wafer", "Z1_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z1_df.append(bad_z1_df), columns=good_z1_df.columns)

    # Remove Unnecessary Columns
    df = remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    print_heatmap(df.corr())

    # Dividing into Inputs and Outputs and run Random Forest Classification
    result = forest_test(df)
    result_list = find_threshold(result, threshold)
    df = pd.DataFrame(result_list, columns=['Type', 'Count'])
    print(df.groupby('Type').max())


def run_z2(threshold):
    # reading all Z1 Data
    good_z2_df = read_data_from_directory("good_wafer", "Z2_100")
    bad_z2_df = read_data_from_directory("bad_wafer", "Z2_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z2_df.append(bad_z2_df), columns=good_z2_df.columns)

    # Remove Unnecessary Columns
    df = remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    print_heatmap(df.corr())

    # Dividing into Inputs and Outputs and run Random Forest Classification
    result = forest_test(df)
    result_list = find_threshold(result, threshold)
    df = pd.DataFrame(result_list, columns=['Type', 'Count'])
    print(df.groupby('Type').max())


def run_z3(threshold):
    # reading all Z1 Data
    good_z3_df = read_data_from_directory("good_wafer", "Z3_100")
    bad_z3_df = read_data_from_directory("bad_wafer", "Z3_100")

    # Creating combined dataset of both good and bad
    df = pd.DataFrame(good_z3_df.append(bad_z3_df), columns=good_z3_df.columns)

    # Remove Unnecessary Columns
    df = remove_columns(df)

    # Configuring the Heatmap to make it easier to see
    print_heatmap(df.corr())

    # Dividing into Inputs and Outputs and run Random Forest Classification
    result = forest_test(df)
    result_list = find_threshold(result, threshold)
    df = pd.DataFrame(result_list, columns=['Type', 'Count'])
    print(df.groupby('Type').max())


if __name__ == "__main__":
    threshold = int(input("Enter the threshold: "))
    run_z1(threshold)
    run_z2(threshold)
    run_z3(threshold)