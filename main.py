# This is a sample Python script.

import dbConnect
import matplotlib.pyplot as plt
import seaborn as sns
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    frame = dbConnect.dataRetrieve()
    print(frame.info())

    # Drop the "First Name" and "Last Name" columns
    frame.drop(columns=['First Name', 'Last Name', 'Position'], inplace=True)
    frame.dropna()

    # Scatter plot for Points vs. Shots
    plt.figure(figsize=(8, 6))
    plt.scatter(frame['Points'], frame['Time On Field'], color='orange')
    plt.title('Time on Field vs. Points')
    plt.xlabel('Points')
    plt.ylabel('Time on Field')
    plt.show()

    #
    # # Heatmap for correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(frame.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix')
    # plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
