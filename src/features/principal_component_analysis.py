import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

class pca():

    def collect_features(self, df, feature_labels, scale=True):
        """gets feature columns from given dataframe and scales data between 0 and 1 per column"""
        self.f_labels = feature_labels
        X = df.loc[:, feature_labels]
        if scale:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            
        self.X = X
        return X

    def run(self):
        """conducts PCA and returns formatted loadings"""

        model = PCA()
        X_pca = model.fit_transform(self.X)
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        self.X_pca = pd.DataFrame(X_pca, columns=component_names)
        
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        
        loadings = pd.DataFrame(
            model.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=self.X.columns,  # and the rows are the original features
        )

        self.loadings = loadings
        self.model = model
        return loadings, model

    def visualize(self):
        """plots a bar chart, cumulative plot, heatmap , and scatter (1st and 2nd) of the pca loadings"""
        fig, axes = plt.subplots(1,3,figsize=(20,5))
        n = self.model.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        my_cmap = plt.get_cmap("Blues")
        rescale = lambda y: y / np.max(y)
        evr = self.model.explained_variance_ratio_
        ax = axes[0]
        ax.bar(grid, evr, edgecolor="black",color=my_cmap(rescale(evr)),zorder=10)
        ax.set_xlabel("Principal Component", fontsize=16)
        ax.set_ylabel("% Explained Variance", fontsize=16)
        ax.set_ylim([0.0, 1.0])
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        # Cumulative Variance
        cv = np.cumsum(evr)
        axes[0].plot(np.r_[0, grid], np.r_[0, cv], "o-", linewidth=2,color="black",zorder=0)
        ax.set_xticklabels(range(-1,len(self.loadings)+1),fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Heatmap
        hm = sns.heatmap(self.loadings,vmin=-1,vmax=1,annot=True,fmt=".1f",square=True,
                    linecolor="black",linewidth=1,cmap="coolwarm_r",cbar_kws={"ticks":[-1,-0.5,0,0.5,1]},ax=axes[1])
        hm.set_yticklabels(self.f_labels,rotation=0,fontsize=14)
        hm.set_xticklabels(["PC1","PC2","PC3","PC4","PC5","PC6"],rotation=0,fontsize=14)
        # PC1 and PC2
        ax = axes[2]
        ax.scatter(self.loadings.iloc[:,0],self.loadings.iloc[:,1],color="black",s=100)
        ax.set_xlabel("1st Principal Component",fontsize=16)
        ax.set_xlim([-1,1])
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_ylabel("2nd Principal Component",fontsize=16)
        ax.set_ylim([-1,1])
        ax.set_yticks([-1,-0.5,0,0.5,1])
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['right'].set_position(('data', 0))
        ax.spines['top'].set_position(('data', 0))
        for loc in ["bottom","left"]:
            ax.spines[loc].set_visible(False)
                
        plt.subplots_adjust(wspace=0.4)
        plt.show()

        return fig, ax

    def compare_loadings(self, loadings1, loadings2, n_components=6):
        """scatters loadings from the specified number of PCs"""
        fig, axes = plt.subplots(1,n_components,figsize=(4*n_components,4),sharey=True)
        for i, ax in enumerate(axes.flat):
            sns.regplot(loadings1.iloc[:,i],loadings2.iloc[:,i],ci=68,ax=ax)
            ax.set_xlim([-1,1])
            ax.set_xlabel("")
            ax.set_ylim([-1,1])
            ax.set_ylabel("")
            r = np.corrcoef(loadings1.iloc[:,i],loadings2.iloc[:,i])
            ax.set_title(f"PC{i+1} r$^2$: {round(r[1,0],2)}",fontsize=14)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            
        ax.set_yticks([-1,-0.5,0,0.5,1])
        plt.show()
        plt.close()