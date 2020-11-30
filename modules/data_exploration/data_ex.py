import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def import_data(my_file):
    """
    used to import the project data into a pandas dataframe
    :param my_file: file name and path as string (csv file expected)
    :return: pandas dataframe
    """
    my_df = pd.read_csv(my_file)
    return my_df


def gen_bulk_data(my_file):
    f"""
    Return a dictionary, which contains different formated dataframes
    :param my_file: file name and path as string (csv file expected)
    :return: dictionary which includes several dictionaries each containing 6 different dataframes
             'x_train' / 'x_test' / 'x_val' / 'y_train' / 'y_test' / 'y_val'
    """
    bulk = {}

    my_df = import_data(my_file)
    
    y = my_df['class']
    x = my_df.drop(['class'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y)

    bulk['original'] = {'x_train': x_train,
                        'y_train': y_train,
                        'x_test': x_test,
                        'y_test': y_test,
                        'x_val': x_val,
                        'y_val': y_val}

    col_drop = ['state',
                'area_code',
                'total_day_charge',
                'total_eve_charge',
                'total_night_charge',
                'total_intl_charge']

    bulk['dropped'] = {'x_train': x_train.drop(columns=col_drop, axis=1),
                       'y_train': y_train,
                       'x_test': x_test.drop(columns=col_drop, axis=1),
                       'y_test': y_test,
                       'x_val': x_val.drop(columns=col_drop, axis=1),
                       'y_val': y_val}

    bulk['scaled'] = {'x_train': scale_data(x_train),
                      'y_train': y_train,
                      'x_test': scale_data(x_test),
                      'y_test': y_test,
                      'x_val': scale_data(x_val),
                      'y_val': y_val}

    bulk['scaled and dropped'] = {'x_train': scale_data(x_train).drop(columns=col_drop, axis=1),
                                  'y_train': y_train,
                                  'x_test': scale_data(x_test).drop(columns=col_drop, axis=1),
                                  'y_test': y_test,
                                  'x_val': scale_data(x_val).drop(columns=col_drop, axis=1),
                                  'y_val': y_val}

    return bulk


def scale_data(my_df):
    """
    Uses a scaler on several columns of the provided dataframe
    :param my_df: pandas dataframe, which needs to be scaled
    :return:  pandas dataframe, with scaled values
    """
    # columns to scale
    sca_columns = ['number_vmail_messages',
                   'total_day_minutes',
                   'total_day_calls',
                   'total_day_charge',
                   'total_eve_minutes',
                   'total_eve_calls',
                   'total_eve_charge',
                   'total_night_minutes',
                   'total_night_calls',
                   'total_night_charge',
                   'total_intl_minutes',
                   'total_intl_calls',
                   'total_intl_charge']

    std = StandardScaler()
    scaled = std.fit_transform(my_df[sca_columns])
    scaled = pd.DataFrame(scaled, columns=sca_columns)
    my_df = my_df.drop(columns=sca_columns, axis=1)
    my_df = my_df.merge(scaled, left_index=True, right_index=True, how="left")

    return my_df


if __name__ == '__main__':
    """
    Test the script
    """
    data_file = '../../data/churn.csv'
    df = gen_bulk_data(data_file)
    for key, value in df.items():
        print('###########')
        print('# {}'.format(key))
        print('###########')
        for x_key, x_value in value.items():
            print('>>>>>>>')
            print('> {}'.format(x_key))
            print('>>>>>>>')
            print(x_value)
