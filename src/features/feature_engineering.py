import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

class fe():
    """
    Methods used to help investigate and build features
    """

    def encode_categoricals(self,feature_set,verbose=False):
        """
        Encodes categorical variables to numeric dtypes

        Inputs:
        - feature_set: dataframe with features as columns

        Returns new dataframe with categorical variables encoded as numbers
        """
        df = feature_set.copy() # so we don't overwrite the original data
        if verbose:
            print("Categorical Variables:")
        for col in df.select_dtypes("object"):
            if verbose:
                print(f"\t{col}")
            df[col], _ = df[col].factorize() # encode the data

        return df

    def get_datasets(self,original_dataset,feature_labels,target_labels,verbose=False):
        """
        Gets the feature and target datasets
        """
        temp = original_dataset.copy()
        if verbose:
            df = self.encode_categoricals(temp,verbose=True)
        else:
            df = self.encode_categoricals(temp,verbose=False)
        
        X = df[feature_labels]
        y = df[target_labels]
        
        return X, y

    def get_mi_scores(self, X, y, discrete_features=False, tolerance=0):
        """
        Gets the Mutual Information (MI) scores
        
        Inputs:
        - X: dataframe of features dataset
        - y: array/series of target
        
        Returns MI scores as a series
        """
        # getting and formatting mi scores for output
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        if tolerance > 0:
            mi_scores = mi_scores[mi_scores >= tolerance]
        mi_scores = mi_scores.sort_values(ascending=False)
        
        return mi_scores

    def plot_mi_scores(self, scores):
        """
        Plots the Mutual Information (MI) scores
        
        Returns void
        """
        # setting up bar chart
        vals = scores.sort_values(ascending=True)
        locs = np.arange(len(vals))
        ticks = list(vals.index)
        formatted_ticks = []
        for tick in ticks:
            formatted_ticks.append(tick.replace("_", " ").title())
        my_cmap = plt.get_cmap("Blues")
        rescale = lambda y: y / np.max(y)
        
        _, ax = plt.subplots(figsize=(8,5))
        ax.barh(locs, vals, color=my_cmap(rescale(vals)), edgecolor="black")
        # formatting x-axis
        try:
            upper_x = max(0.5,max(vals))
        except:
            upper_x = 0.5

        ax.set_xlim([0,upper_x])
        # formatting y-axis
        plt.yticks(locs, formatted_ticks)
        # formatting remainder
        ax.set_title("Mutual Information Scores")
        
        for spine_loc in ["top","right"]:
            ax.spines[spine_loc].set_visible(False)
        
        plt.show()
        plt.close()

    def plot_high_scoring_relationships(self, X, y, mi_scores, num_scores=3, width=16):
        """
        Plots scatterplots of the top-ranking features and the given target
        
        Inputs:
        - X: dataframe of features
        - y: series of targets
        - mi_scores: series/list of MI scores
        - num_scores: integer of number of variables/scores to show
        - width: width of figure which also controls the height (width/num_scores)
        
        Returns void
        """
        try:
            target = y.name
        except AttributeError:
            target = y.columns[0]
        df = X.merge(right=y,left_index=True,right_index=True)
        _, axes = plt.subplots(1,num_scores,figsize=(num_scores*5,5))
        colors = cm.get_cmap('Blues_r', num_scores)(range(num_scores))
        for var, score, color, ax in zip(list(mi_scores.index[:num_scores]),mi_scores,colors,axes.flat):
            sns.scatterplot(x=var, y=target, data=df, color=color, edgecolor="black",ax=ax)
            # formatting x
            ax.set_xlabel(var.replace("_", " ").title())
            # formatting y
            ax.set_ylabel(target.replace("_", " ").title())
            # formatting remainder
            ax.set_title(f"MI Score: {round(score,3)}")
            for spine_loc in ["top","right"]:
                ax.spines[spine_loc].set_visible(False)
                
        plt.show()
        plt.close()

    def check_features_against_targets(self, df, target_labels, features_to_show=3, tolerance=0.01, verbose=False):
        """
        Checks the features in df to targets given by target_labels which are also in df

        Inputs:
        - df:
        - target_labels:

        Returns void
        """
        temp = df.copy()
        for target_label in target_labels:
            if verbose:
                print(f"Target: {target_label.replace('_',' ').title()}")
            features = [feature for feature in temp.columns if feature not in target_labels]
            # getting data
            try:
                X, y = self.get_datasets(original_dataset=temp,feature_labels=features,target_labels=target_label)
            except KeyError:
                print(f"\t{target_label} not in dataframe")
                continue
            # getting MI scores
            mi_scores = self.get_mi_scores(X, y, tolerance=tolerance)
            if len(mi_scores) > 0:
                # plotting scores
                self.plot_mi_scores(mi_scores)
                # scattering strong relationships
                self.plot_high_scoring_relationships(X, y, mi_scores, num_scores=features_to_show)
                return True
            else:
                return False