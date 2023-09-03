"""

Created By: Thomas Haskell
Date: Summer 2023

===== Machine Learning Project =====
PLL Player Year-to- Year Stat Prediction

"""
import pandas as pd

import dbConnect
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import tensorflow_decision_forests as tfdf

import seaborn as sns
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

t23 = tf.constant([])
t22 = tf.constant([])
t21 = tf.constant([])

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    frame23 = dbConnect.dataRetrieve("stats2023")
    frame22 = dbConnect.dataRetrieve("stats2022")
    frame21 = dbConnect.dataRetrieve("stats2021")
    collected_years = [frame21, frame22, frame23]
    years_label = ['2021', '2022', '2023']

    tensorYears = [t21, t22, t23]

# Mapping players to a unique identifier to interpret predictions
    names_mapped = {player: idx for idx, player in enumerate(frame23['last_name'].unique())}
    player_identifiers = pd.DataFrame({'last_name': frame23['last_name'].unique(), 'player_id': range(len(names_mapped))})

    # Drop the "First Name" and "Last Name" columns
    for year in collected_years:
        # Gets rid of non-numeric columns which have no sense in analyzing
        year.drop(columns=['first_name', 'last_name', 'position', 'short_handed_goals_against'], inplace=True)
        # Convert data types to float64
        year = year.astype(float)

    # # Scatter plot for Points vs. Shots
    # plt.figure(figsize=(8, 6))
    # plt.scatter(frame['Points'], frame['Time On Field'], color='orange')
    # plt.title('Time on Field vs. Points')
    # plt.xlabel('Points')
    # plt.ylabel('Time on Field')
    # plt.show()

    # # Heatmap for correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(frame.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix')
    # plt.show()

    # ----- TensorFlow Manipulation ----- #
    # Create tensorflow datasets
    print(frame23[:5])
    tf_dataset_2023 = tfdf.keras.pd_dataframe_to_tf_dataset(frame23, label="points")

    tf_dataset_2022 = tfdf.keras.pd_dataframe_to_tf_dataset(frame22, label="points")

    # Combine previous years data for training
    training_df = pd.concat([frame21, frame22], ignore_index=True)  # Merges dataframes w reset index
    tf_combinedYears_train = tfdf.keras.pd_dataframe_to_tf_dataset(training_df, label="points")

    # Initialize and train model
    model = tfdf.keras.RandomForestModel(verbose=0)
    model.fit(tf_combinedYears_train)

    # Create serving dataset
    tf_serving_dataset = tf_dataset_2023

    # Predict the target stat for the year
    predictions = model.predict(tf_serving_dataset)
    filtered_predictions = predictions[player_identifiers.index]

    print(filtered_predictions[4])
    predicted_points = filtered_predictions[:, 4] # Specifying points column
    print("Length of player_identifiers:", len(player_identifiers))
    print("Length of predictions:", len(predictions))
    print(player_identifiers.shape)
    print(filtered_predictions.shape)

    # Combines player identifiers with predicted stats
    final_results = pd.DataFrame({'player_id': player_identifiers.index, 'predicted_stat': predicted_points})
    final_results = pd.merge(final_results, player_identifiers, on='player_id')

    print(final_results.sort_values(by='predicted_stat', ascending=False))

    # MAKES DATAFRAMES INTO ARRAYS AND THEN EACH YEAR INTO INDIVIDUAL TENSORS
    # for year, tensor in zip(collected_years, tensorYears):
    #     tensor = tf.constant(year.to_numpy(), dtype=tf.float64)
    #     print("BOOOOOM")
    #     print(tensor)
    #
    #

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
