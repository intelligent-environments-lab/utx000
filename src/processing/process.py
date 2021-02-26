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

    def plot_missing_data(self,df):
        """
        Plots the number of missing values

        Returns void
        """
        missing_vals = df.isnull().sum()
        vals = missing_vals[missing_vals > 0].sort_values()
        # setting up bar chart
        locs = np.arange(len(vals))
        ticks = list(vals.index)
        formatted_ticks = []
        for tick in ticks:
            formatted_ticks.append(tick.replace("_", " ").title())
        my_cmap = plt.get_cmap("Blues")
        rescale = lambda y: y / np.max(y)

        _, ax = plt.subplots(figsize=(8,5))
        rects = ax.barh(locs, vals, color=my_cmap(rescale(vals)), edgecolor="black")
        # formatting y-axis
        plt.yticks(locs, formatted_ticks)
        # formatting remainder
        n = len(df)
        ax.set_title(f"Missing Values - Total Values: {n}")

        for spine_loc in ["top","right"]:
            ax.spines[spine_loc].set_visible(False)
            
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                width = rect.get_width()
                ax.annotate('{}'.format(width),
                            xy=(width, rect.get_y()),
                            xytext=(3, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='left', va='bottom')
                
        autolabel(rects)

        plt.show()
        plt.close()

    def plot_correlation_matrix(self,df,annotate=True):
        """
        Plots correlation matrix between all variables in the df
        
        Inputs:
        - df: dataframe with columns named for the varaiables
        
        """
        corr = df.corr()
        corr = round(corr,2)
        #mask = np.triu(np.ones_like(corr, dtype=bool))
        _, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr,
                        vmin=-1, vmax=1, center=0, 
                        cmap=sns.diverging_palette(20, 220, n=200),cbar_kws={'ticks':[-1,-0.5,0,0.5,1]},
                        square=True,linewidths=0.5,linecolor="black",annot=annotate,ax=ax)

        yticklabels = ax.get_yticklabels()
        yticklabels[0] = ' '
        ax.set_yticklabels(yticklabels,rotation=0,ha='right')

        xticklabels = ax.get_xticklabels()
        xticklabels[-1] = ' '
        ax.set_xticklabels(xticklabels,rotation=-45,ha='left')
            
        plt.show()
        plt.close()
