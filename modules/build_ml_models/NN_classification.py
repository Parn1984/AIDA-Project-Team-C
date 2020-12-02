"""
+++ AUTHOR: Tobias Habermann +++

Neuronal Netwokrs classification helper functions

Methods:
    build_model     : set up a neural network with the given parameters, hidden layer size can be defined via list
    build_model_seq : set up a neural network with the given parameters with keras Seuential, hidden layer size can be defined via list

"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# function for model-set-up

def build_model(input_shape=20, hidden_shapes=[32], hidden_activation='relu',
                opt='adam', dropout=0.1, loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):
    # INPUT layer
    input_lay = Input(shape=(input_shape,), name='input')

    # first HIDDEN layer
    if len(hidden_shapes) > 0:
        print(hidden_shapes[0])
        x = Dense(hidden_shapes[0], activation=hidden_activation, name='hidden_dense_1')(input_lay)
        # add Dropout layer
        if dropout > 0:
            x = Dropout(dropout)(x)

    # other HIDDEN layers
    for i, val in enumerate(hidden_shapes[1:]):
        x = Dense(val, activation=hidden_activation, name=f'hidden_dense_{i+2}')(x)

    # OUTPUT layer
    output = Dense(1, activation='sigmoid', name='dense_1')(x)


    model = Model(input_lay, output, name="DENSE_CHURN_PREDICTION")

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model




# function for model-set-up Sequential
def build_model_seq(input_shape=20, hidden_shapes=[32], hidden_activation='relu',
                    opt='adam', dropout=0.0, loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]):

    model = Sequential()

    # INPUT layer
    model.add(Input(shape=(input_shape,), name='input_layer'))
    # HIDDEN Layers
    for i, val in enumerate(hidden_shapes):
        model.add(Dense(val, activation=hidden_activation, name=f'hidden_layer_{i}'))
        model.add(Dropout(dropout))

    # OUTPUT layer
    model.add(Dense(1,  activation="sigmoid", name='output_layer'))

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


# Plot a nice confusion matrix
def plot_confusion_matrix(y_test, y_pred, cmap='RdYlGn', normalize='true'):

    if normalize == 'true':
        format = '.4f'
    else:
        format = 'd'

    # calculate confusion matrix
    data = confusion_matrix(y_test, y_pred, normalize=normalize)
    df_cm = pd.DataFrame(data, columns=['no churn', 'churn'], index=['no churn', 'churn'])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(5, 3))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap=cmap, annot=True, annot_kws={"size": 14}, fmt=format)
    plt.title("Confusion Matrix", fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')


def print_classification_report(y_test, y_pred):
    print("\nClassification report:\n", classification_report(y_test, y_pred))