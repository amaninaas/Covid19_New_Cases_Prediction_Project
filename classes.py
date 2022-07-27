# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
import numpy as np



class EDA:
    def lineplot_graph(self,con_col,df):
        '''
         This is used to plot continous data

        Parameters
        ----------
        con_col : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Continous Visualization
        for i in con_col: 
            plt.figure()
            plt.plot(df[i])
            plt.show()

class ModelDevelopment:
    def simple_dl_model(self,input_shape,nb_class,nb_node,dropout_rate,
                        activation):
        '''
         This is one layer model using only LSTM, Dense and Dropout

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        nb_class : TYPE
            DESCRIPTION.
        nb_node : TYPE
            DESCRIPTION.
        dropout_rate : TYPE
            DESCRIPTION.
        activation : TYPE
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(LSTM(nb_node))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation))
        model.summary()
        
        return model

class MAPE:
    def mape(self,actual_cases, predicted_cases): 
        actual_cases, predicted_cases= np.array(
            actual_cases),np.array(predicted_cases)
        return np.mean(np.abs((actual_cases - predicted_cases) / 
                              actual_cases)) * 100


class ModelEvaluation:
    def plot_hist_graphy(self,hist):
        plt.figure()
        plt.plot(hist.history['mean_absolute_percentage_error'])
        plt.plot(hist.history['val_mean_absolute_percentage_error'])
        plt.xlabel('epoch')
        plt.legend(['Training MAPE','Validation MAPE'])
        plt.show()


    def plot_line_graph(self,y_test, predicted_cases):
        plt.figure()
        plt.plot(y_test,color='red')
        plt.plot(predicted_cases,color='blue')
        plt.xlabel('Time')
        plt.ylabel('Cases')
        plt.legend(['Actual','Predicted'])
        plt.show()

    def inverse_line_graph(self,actual_cases, predicted_cases):
        plt.figure()
        plt.plot(actual_cases,color='red')
        plt.plot(predicted_cases,color='green')
        plt.xlabel('Days')
        plt.ylabel('Cases')
        plt.legend(['Actual','Predicted'])
        plt.show()