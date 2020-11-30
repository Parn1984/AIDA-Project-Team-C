import os
import matplotlib.pyplot as plt
import pandas as pd


def import_data(data_file):
    # TODO: schreib mal
    my_df = pd.read_csv(data_file)
    return my_df

def view_data(data):
    # TODO: schreib mal
    plt.plot(data)


if __name__ == '__main__':
    data_file = '../../data/churn.csv'
    df = import_data(data_file)
    print(df)