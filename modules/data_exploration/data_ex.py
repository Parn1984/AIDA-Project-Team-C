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


def gen_xy(my_file):
    """
    loads data and split it into X and y
    :param my_file: file name and path as string (csv file expected)
    :return: x dataframe and y series
    """
    my_df = import_data(my_file)

    y = my_df['class']
    x = my_df.drop(['class'], axis=1)
    return x, y


def gen_bulk_data(my_file, keep=None):
    f"""
    Return a dictionary, which contains different formated dataframes 
    :param my_file: file name and path as string (csv file expected)
    :param keep: optional list of columns to keep, when dropping.
                 Default List of dropped columns when empty: 
                 ['state',
                  'area_code',
                  'phone_number',
                  'total_day_charge',
                  'total_eve_charge',
                  'total_night_charge',
                  'total_intl_charge']
    :return: dictionary which includes several dictionaries each containing 6 different dataframes
             'x_train' / 'x_test' / 'x_val' / 'y_train' / 'y_test' / 'y_val'
    """
    if keep is None:
        keep = []
    bulk = {}

    x, y = gen_xy(my_file)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    bulk['original'] = {'x_train': x_train,
                        'y_train': y_train,
                        'x_test': x_test,
                        'y_test': y_test,
                        'x_val': x_val,
                        'y_val': y_val}

    col_drop = ['state',
                'area_code',
                'phone_number',
                'total_day_charge',
                'total_eve_charge',
                'total_night_charge',
                'total_intl_charge']

    if len(keep) > 0:
        for entry in keep:
            if entry in col_drop:
                col_drop.remove(entry)

    bulk['dropped'] = {'x_train': x_train.drop(columns=col_drop, axis=1),
                       'y_train': y_train,
                       'x_test': x_test.drop(columns=col_drop, axis=1),
                       'y_test': y_test,
                       'x_val': x_val.drop(columns=col_drop, axis=1),
                       'y_val': y_val}

    x_train_scaled, x_test_scaled, x_val_scaled = scale_data(x_train, x_test, x_val)
    bulk['scaled'] = {'x_train': x_train_scaled,
                      'y_train': y_train,
                      'x_test': x_test_scaled,
                      'y_test': y_test,
                      'x_val': x_val_scaled,
                      'y_val': y_val}

    x_train_enc = enc(x_train)
    x_test_enc = enc(x_test)
    x_val_enc = enc(x_val)
    bulk['encoded'] = {'x_train': x_train_enc,
                       'y_train': y_train,
                       'x_test': x_test_enc,
                       'y_test': y_test,
                       'x_val': x_val_enc,
                       'y_val': y_val}

    x_train_enc_sca, x_test_enc_sca, x_val_enc_sca = scale_data(x_train_enc, x_test_enc, x_val_enc)
    bulk['encoded and scaled'] = {'x_train': x_train_enc_sca,
                                  'y_train': y_train,
                                  'x_test': x_test_enc_sca,
                                  'y_test': y_test,
                                  'x_val': x_val_enc_sca,
                                  'y_val': y_val}

    bulk['scaled and dropped'] = {'x_train': x_train_scaled.drop(columns=col_drop, axis=1),
                                  'y_train': y_train,
                                  'x_test': x_test_scaled.drop(columns=col_drop, axis=1),
                                  'y_test': y_test,
                                  'x_val': x_val_scaled.drop(columns=col_drop, axis=1),
                                  'y_val': y_val}

    return bulk


def enc(my_df):
    state_enc = pd.get_dummies(data=my_df['state'], prefix='state')
    area_enc = pd.get_dummies(data=my_df['area_code'], prefix='area')
    encoded = state_enc.merge(area_enc, left_index=True, right_index=True, how="left")
    df_enc = my_df.drop(columns=['state', 'area_code'], axis=1)
    df_enc = df_enc.merge(encoded, left_index=True, right_index=True, how="left")

    return df_enc


def scale_data(my_x_train, my_x_test, my_x_val):
    """
    Uses a scaler on several columns of the provided dataframe
    :param my_x_val:
    :param my_x_test:
    :param my_x_train:
    :return:  pandas dataframe, with scaled values
    """

    std = StandardScaler()

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

    std.fit(my_x_train[sca_columns])

    my_x_train_scaled = scale(my_x_train, std, sca_columns)
    my_x_test_scaled = scale(my_x_test, std, sca_columns)
    my_x_val_scaled = scale(my_x_val, std, sca_columns)

    return my_x_train_scaled, my_x_test_scaled, my_x_val_scaled


def scale(my_set, scaler, my_sca_columns):
    """
    Uses provided scaler
    :param my_set: dataframe to be scaled
    :param scaler: scaler to be used
    :param my_sca_columns: dataframe columns to scale
    :return: provided dataframe, with scaled columns
    """

    scaled = scaler.transform(my_set[my_sca_columns])
    scaled = pd.DataFrame(scaled, columns=my_sca_columns)
    return_df = my_set.drop(columns=my_sca_columns, axis=1)
    return_df = return_df.merge(scaled, left_index=True, right_index=True, how="left")
    return return_df


def get_bulk_inst(my_bulk, instance):
    """
    :param my_bulk: bulk dictionary as provided by gen_bulk_data()
    :param instance: requested instance of the bulk dictionary
    :return: x_train, x_test, x_val, y_train, y_test, y_val
    """
    x_train = my_bulk[instance]['x_train']
    x_test = my_bulk[instance]['x_test']
    x_val = my_bulk[instance]['x_val']
    y_train = my_bulk[instance]['y_train']
    y_test = my_bulk[instance]['y_test']
    y_val = my_bulk[instance]['y_val']

    return x_train, x_test, x_val, y_train, y_test, y_val


if __name__ == '__main__':
    """
    Test the script
    """
    data_file = '../../data/churn.csv'
    df = gen_bulk_data(data_file)
    print(df.keys())

    for key, value in df.items():
        print('###########')
        print('# {}'.format(key))
        print('###########')
        for x_key, x_value in value.items():
            print('>>>>>>>')
            print('> {}'.format(x_key))
            print('>>>>>>>')
            print(x_value)
