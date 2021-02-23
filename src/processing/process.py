import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import matplotlib.dates as mdates

from scipy import stats
from sklearn.cluster import KMeans

class preprocess():

    def __init__(self):
        pass

    def plot_hist(self, df):
        """
        Plots histograms of each of the columns in the given dataframe
        
        Inputs:
        - df: dataframe containing columns with numeric data
        
        Returns void
        """
        _, axes = plt.subplots(1,len(df.columns),figsize=(5*len(df.columns),5),sharey=True)
        colors = cm.get_cmap('Blues_r', len(df.columns))(range(len(df.columns)))
        if len(df.columns) > 1:
            for col, color, ax in zip(df.columns,colors,axes.flat):
                ax.hist(df[col],color=color,rwidth=0.85,edgecolor="black",)
                ax.set_title(col.replace("_"," ").title())
                ax.set_xlim(xmin=0)
                plt.subplots_adjust(wspace=0.1)
                
                for loc in ["right","top"]:
                    ax.spines[loc].set_visible(False)
        else:
            axes.hist(df.iloc[:,0],color=colors,rwidth=0.85,edgecolor="black",)
            axes.set_title(df.columns[0].replace("_"," ").title())
            axes.set_xlim(xmin=0)
            plt.subplots_adjust(wspace=0.1)
            
            for loc in ["right","top"]:
                axes.spines[loc].set_visible(False)
            
        plt.show()
        plt.close()

    def normalize(self, df, method="boxcox"):
        """
        Normalizes each of the columns in the given dataframe according to the method
        
        Inputs:
        - df: dataframe with numeric columns
        - method: string specifying the normalization method
        
        Returns dataframe with normalized columns
        """
        if method == "boxcox":
            for column in df.columns:
                df[column] = stats.boxcox(df[column])[0]
        elif method == "log":
            for column in df.columns:
                df[column] = np.log(df[column])
        elif method == "yeojohnson":
            for column in df.columns:
                df[column] = stats.yeojohnson(df[column])[0]
        return df
