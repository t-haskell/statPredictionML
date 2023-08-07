'''

Created By: Thomas Haskell
Date: Summer 2023

===== Machine Learning Project =====
PLL Player Year-to- Year Stat Prediction

'''

import dbConnect
import matplotlib.pyplot as plt
import tensorflow as tf
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
    print_hi('PyCharm')
    frame23 = dbConnect.dataRetrieve("stats2023")
    frame22 = dbConnect.dataRetrieve("stats2022")
    frame21 = dbConnect.dataRetrieve("stats2021")
    collected_years = [frame21, frame22, frame23]
    tensorYears = [t21, t22, t23]

    print(frame23.info())
    print(frame22.info())
    print(frame21.info())
    print("Before drop")


    # Drop the "First Name" and "Last Name" columns
    for year in collected_years:
        # Gets rid of non-numeric columns which have no sense in analyzing
        year.drop(columns=['First Name', 'Last Name', 'Position', 'Short Handed Goals Against'], inplace=True)
        print(year.info())
        # Convert data types to float64
        year = year.astype(float)
        print(year.info())


    # # Scatter plot for Points vs. Shots
    # plt.figure(figsize=(8, 6))
    # plt.scatter(frame['Points'], frame['Time On Field'], color='orange')
    # plt.title('Time on Field vs. Points')
    # plt.xlabel('Points')
    # plt.ylabel('Time on Field')
    # plt.show()

    #
    # # Heatmap for correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(frame.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix')
    # plt.show()

    ####### TensorFlow Manipulation ######
    for year, tensor in zip(collected_years, tensorYears):
        # Loop makes DataFrames into arrays and turns them into individual Tensors
        tensor = tf.constant(year.to_numpy(), dtype=tf.float64)
        print("BOOOOOM")
        print(tensor)


    # arr23 = frame23.to_numpy()
    # arr22 = frame22.to_numpy()
    # arr21 = frame21.to_numpy()
    # pll2023 = tf.constant(playerArr, dtype=tf.float64)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
