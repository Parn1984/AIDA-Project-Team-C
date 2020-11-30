import os
import matplotlib.pyplot as plt
import pandas as pd

data_file = '../../data/churn.csv'

def import_data():
    # TODO: schreib mal
    my_df = pd.read_csv(data_file)
    return my_df

def view_data(data):
    # TODO: schreib mal
    plt.plot(data)


if __name__ == '__main__':
    df = import_data()
    print(df)