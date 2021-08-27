import pandas as pd
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ML Supporting
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score,log_loss

# Supporting
from datetime import datetime, timedelta

class ImportProcessing():

    def __init__(self,data_dir="../../data/"):
        """
        Imports all the raw data
        """
        self.data_dir = data_dir
        # DataFrames
        self.mood_only = pd.read_csv(f"{self.data_dir}interim/mood_prediction/beiwe-beiwe-ema_morning-ema_evening.csv",index_col=0,
            parse_dates=["timestamp_e","timestamp_m","date"],infer_datetime_format=True)
        self.mood_and_activity = pd.read_csv(f"{self.data_dir}interim/mood_prediction/fitbit-beiwe-beiwe-activity-ema_morning-ema_evening.csv",index_col=0,
            parse_dates=["timestamp_e","timestamp_m","date"],infer_datetime_format=True)
        

        self.dfs = (self.mood_only,self.mood_and_activity)
        # Cleaning
        for df in self.dfs:
            self.replace_str(df)
            self.convert_to_numeric(df,[c for c in df.columns if c.endswith("_e") or c.endswith("_m")])
            self.drop_columns(df,["beacon","timestamp_e","timestamp_m"])

    def replace_str(self, df):
        """
        Replaces all instances of string responses with the correct numerical value
        """
        for response, value in zip(["Very much"],[3]):
            df.replace(response,value,inplace=True)

    def convert_to_numeric(self, df, columns):
        """
        Converts the given columns in the dataframe to numeric value types
        """
        for col in columns:
            df[col] = pd.to_numeric(df[col])

    def drop_columns(self,df, columns):
        """
        Drops the specified columns from the provided dataframe
        """
        for col in columns:
            try:
                df.drop(col,axis="columns",inplace=True)
            except KeyError:
                print(f"Column {col} not in DataFrame")

    def remove_participant(self, df_in, pt, by_id="beiwe"):
        """
        Remove participant from the given datasets
        """
        df = df_in.copy()
        df = df[df[by_id] != pt]
        return df

class Inspection():

    def __init__(self):
        pass

    def get_mood_distribution(self,df_in,moods=["content","stress","lonely","sad","energy"],plot=False):
        """
        Parameters
        ----------
        df_in : DataFrame
            Original data with columns corresponding to the provided moods
        moods : list-like, default ["content","stress","lonely","sad","energy"]
            Strings of the moods to consider - must be columns in df_in
        plot : boolean
            whether or not to output the histograms of the scores
            
        Returns
        -------
        df : DataFrame
            
        """
        res = {mood: [] for mood in moods}
        df = df_in.copy()
        for timing in ["_m","_e"]:
            for mood in moods:
                df[f"{mood}{timing}"] = pd.to_numeric(df[f"{mood}{timing}"])
        if plot:
            _, axes = plt.subplots(2,len(moods),figsize=(len(moods)*4,6),sharey="row",sharex="col")
            for r, timing, label in zip(range(2),["_m","_e"],["Morning (Features)","Evening (Targets)"]):
                for mood, ax in zip(moods,axes[r,:]):
                    dist = df[f"{mood}{timing}"].value_counts()
                    ax.bar(dist.index,dist.values/dist.sum(),edgecolor="black",color="lightgray")
                    for score, height in zip(dist.index,dist.values/dist.sum()):
                        ax.text(score,height+0.05,round(height*100,1),ha="center")
                    ax.set_ylim([0,1])
                
                    if r == 1:
                        ax.set_xlabel(mood.title(),fontsize=12)
                        
                    # appending results to output
                    res[mood].append(dist.values)
                                    
                axes[r,0].set_ylabel(label)
                
            plt.show()
            plt.close()
        
        return pd.DataFrame(data=res)

    def get_mood_difference(self,df_in,moods=["content","stress","lonely","sad","energy"],plot=False):
        """
        Calculates the score difference between mood scores ofthe same name.
        
        Parameters
        ----------
        df_in : DataFrame
            Original data with columns corresponding to the provided moods
        moods : list-like, default ["content","stress","lonely","sad","energy"]
            Strings of the moods to consider - must be columns in df_in
        plot : boolean
            whether or not to output the histograms of the differences for each
            mood
            
        Returns
        -------
        df : DataFrame
            Original dataset with the mood scores removed and replaced with the 
            differences
        """
        df = df_in.copy()
        for mood in moods:
            df[f"{mood}_diff"] = df[f"{mood}_e"] - df[f"{mood}_m"]
            
        df.drop([col for col in df.columns if col[-2:] in ["_e","_m"]],axis="columns",inplace=True)
        
        if plot:
            _, axes = plt.subplots(1,len(moods),figsize=(len(moods)*5,3),sharey=True)
            for i, (mood, ax) in enumerate(zip(moods,axes)):
                temp = pd.DataFrame(data=df[f"{mood}_diff"].value_counts())
                temp["percent"] = temp[f"{mood}_diff"] / temp[f"{mood}_diff"].sum()
                temp.sort_index(ascending=False,inplace=True)
                rects = ax.bar(temp.index,temp["percent"],edgecolor="black",color="lightgray")
                # x-axis
                if i == 4:
                    ax.set_xlim([-4.5,4.5])
                    ax.set_xticks(np.arange(-4,5,1))
                else:
                    ax.set_xlim([-3.5,3.5])
                    ax.set_xticks(np.arange(-3,4,1))
                if i == 2:
                    ax.set_xlabel("Score Difference (E - M)",fontsize=14)
                # y-axis
                ax.set_ylim([0,1])
                # remainder
                ax.set_title(mood.title() + f" (n={len(df)})")
                for score, height in zip(temp.index,temp["percent"]):
                    ax.text(score,height+0.05,round(height*100,1),ha="center")
                
                
            axes[0].set_ylabel("Percent")
            plt.show()
            plt.close()
            
        return df

    def get_response_n(self,df_in,moods=["content","stress","lonely","sad","energy"],timing="e",plot=False):
        """
        Gets the number of responses from each individual for each provided mood
        
        Parameters
        ----------
        df_in : DataFrame
            Original data with columns corresponding to the provided moods
        moods : list-like, default ["content","stress","lonely","sad","energy"]
            Strings of the moods to consider - must be columns in df_in
        timing : {"e","m"}, default "e"
            specifies whether to consider evening ("e") or morning ("m") EMAs
        plot : boolean
            whether or not to output the histograms of the differences for each
            mood
            
        Returns
        -------
        responses : DataFrame
            DataFrame indexed by participant with columns corresponding to the number
            of responses for the provided moods
        """
        df = df_in.copy()
        responses = {key: [] for key in ["beiwe"] + moods}
        for pt in df["beiwe"].unique():
            responses["beiwe"].append(pt)
            df_pt = df[df["beiwe"] == pt]
            for mood in moods:
                responses[mood].append(len(df_pt[f"{mood}_{timing}"].dropna()))
                
        responses = pd.DataFrame(data=responses)
        if plot:
            _, axes = plt.subplots(len(moods),1,figsize=(22,len(moods)*3),sharex=True,gridspec_kw={"hspace":0.1})
            for mood, ax in zip(moods, axes):
                responses.sort_values(mood,inplace=True)
                ax.scatter(responses["beiwe"],responses[mood],color="black")
                ax.set_title(mood,fontsize=16,pad=0)
                ax.set_ylim([0,70])
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
                
            ax.set_xticklabels(responses["beiwe"].unique(),rotation=-30,ha="left")
            plt.show()
            plt.close()
            
        return responses

class Model():

    def __init__(self):
        self.model_params = {
            "random_forest": {
                "model":RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators":[10,50,100],
                    "max_depth":[1,2,3,4,5],
                    "min_samples_split":[2,4],
                    "min_samples_leaf":[1,2],
                }
            },
            "svc": {
                "model": SVC(),
                "params": {
                    "kernel":["linear","poly","sigmoid","rbf"],
                }
            },
            "gradientboost":{
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators":[10,50,100],
                    "max_depth":[1,2,3,4,5],
                    "min_samples_split":[2,4],
                    "min_samples_leaf":[1,2],
                }
            },
            "logistic_regression": {
                "model":LogisticRegression(random_state=42,max_iter=500),
                "params": {
                    "fit_intercept":[True,False],
                    "solver":["lbfgs","liblinear"],
                }
            },
        }

    def get_x_and_y(self, df_in, mood="content", include_evening=False, additional_features=[]):
        """
        Gets the feature and target datasets corresponding to the target
        
        Parameters
        ----------
        df_in : DataFrame
            Original data with columns corresponding to the provided moods
        mood : {"content","stress","lonely","sad","energy"}, default "content"
            Mood to consider - must be a column in df_in
        include_evening : boolean
            Whether or not to include the other evening mood scores
            
        Returns
        -------
        X : array of floats
            Morning EMA mood reports
        y : list of floats
            Evening EMA mood reports corresponding to the input mood
        groups : Series of strings
            the participants that the X and y vectors are associated with
        """
        df = df_in.copy()
        # removing NaN
        df.dropna(subset=[f"{mood}_e"],axis="rows",inplace=True)
        features = [col for col in df.columns if col.endswith("_m")] + additional_features
        df.dropna(subset=features,axis="rows",inplace=True)
        # getting features and targets
        if include_evening:
            df.dropna(subset=[col for col in df.columns if col.endswith("_e")],axis="rows",inplace=True)
            features += [col for col in df.columns if col.endswith("_e")]
            X = df[features]
            X.drop(f"{mood}_e",axis="columns",inplace=True)
        else:
            X = df[features]
        y = df[f"{mood}_e"].values
        # getting groups
        groups = df["beiwe"]
        return X.values, y, groups

    def binarize_mood(self, df_in, moods=["content","stress","lonely","sad","energy"],binarize_features=True):
        """
        Binarizes mood targets and/or features
        
        Parameters
        ----------
        df_in : DataFrame
            Original data with columns corresponding to the provided moods
        moods : list-like, default ["content","stress","lonely","sad","energy"]
            Strings of the moods to consider - must be columns in df_in
        binarize_features : boolean, default True
            Whether or not to binarize features in addition to targets
        
        Returns
        -------
        df : DataFrame
            original dataframe with mood scores replace with binary values
        """
        df = df_in.copy()
        if binarize_features:
            timings = ["m","e"]
        else:
            timings = ["e"]
        for mood in moods:
            for timing in timings:
                if mood in ["content","energy"]:
                    df[f"{mood}_{timing}"] = [0 if score < 2 else 1 for score in df[f"{mood}_{timing}"]]
                else:
                    df[f"{mood}_{timing}"] = [0 if score == 0 else 1 for score in df[f"{mood}_{timing}"]]
        
        return df

    def binarize_steps(self,df_in,step_goal=10000):
        """
        Converts steps metrics to binary values
        """
        df = df_in.copy()
        if {"steps","active_percent"}.issubset(df.columns):
            df = df[df["active_percent"] > 0.5]
            df["steps"] = df["steps"] / df["active_percent"]
            df.drop("active_percent",axis="columns",inplace=True)
            df["step_goal"] = [1 if steps > step_goal else 0 for steps in df["steps"]]
        else:
            print("Returning orignal dataframe - missing necessary column(s)")

        return df

    def optimize_models(self, df,params,moods=["content","stress","lonely","sad","energy"],additional_features=[]):
        """
        Runs GridSearch to determine the best hyperparameters for the given models
        
        Parameters
        ----------
        df : DataFrame
            data with columns for each mood
        params : dict
            ML models and the parameters that we wish to tune
        moods : list-like of default ["content","stress","lonely","sad","energy"]
            moods to consider for the cross-validation - must be column(s) in df
        
        Returns
        -------
        <results> : DataFrame
            
        """
        scores = []
        for mood in moods:
            for model_name, mp in params.items():
                s = datetime.now()
                print(f"\t{model_name.replace('_',' ').title()}")
                clf = GridSearchCV(mp["model"],mp["params"],cv=5,return_train_score=False)
                X, y, _ = self.get_x_and_y(df,mood=mood,additional_features=additional_features)
                clf.fit(X, y)
                scores.append({
                    "mood":mood,
                    "model":model_name,
                    "best_score":clf.best_score_*100,
                    "best_params":clf.best_params_
                })
                print("\t\tElapsed Time:\t", datetime.now() - s)
                print(f"\t\tBest Score:\t{round(clf.best_score_*100,1)}\n\t\tBest Params:\t{clf.best_params_}")

        return pd.DataFrame(scores,columns=["mood","model","best_score","best_params"])

    def cross_validate(self, df,models,cv_label="skf",moods=["content","stress","lonely","sad","energy"],n_splits=5,verbose=False,additional_features=[]):
        """
        Runs various cross-validation techniques on the provided models
        
        Parameters
        ----------
        df : DataFrame
        models : dict
            sklearn ML models to consider with keys corresponding to the string of the model name and the keys of the sklearn
            model with provided hyperparameters
        cv_label : {"skf","gkf","logo"}, default "skf"
            Specifies the type of cross-validation technique:
            "skf":stratified k-fold, "gkf":group k-fold, "logo":leave one group out
        moods : list-like of default ["content","stress","lonely","sad","energy"]
            moods to consider for the cross-validation - must be column(s) in df
        n_splits : int, default 5
            number of splits to perform for k-fold cross-validation techniques
        verbose : boolean, default False
            verbose mode for debugging primarily leave one group out cross-validation
        
        Returns
        -------
        <results> : DataFrame
            Cross-validation scores from each split and the average per each mood and model
        """
        if cv_label == "skf":
            cv = StratifiedKFold(n_splits=n_splits)
            groups = None # for consistency
            res = {key: [] for key in ["mood","model"] + [f"split_{i+1}" for i in range(n_splits)] + ["mean"]}
        elif cv_label == "gkf":
            cv = GroupKFold(n_splits=n_splits)
            res = {key: [] for key in ["mood","model"] + [f"split_{i+1}" for i in range(n_splits)] + ["mean"]}
        elif cv_label == "logo":
            cv = LeaveOneGroupOut()
            X,y,groups = self.get_x_and_y(df,additional_features=additional_features)
            if verbose:
                for train_index, test_index in cv.split(X, y, groups):
                    print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = groups.iloc[train_index], groups.iloc[test_index]
                    print(X_train, X_test)
            res = {key: [] for key in ["mood","model"] + [f"{pt} ({n})" for pt, n in zip(groups.unique(),groups.value_counts().sort_index())] + ["mean"]}
        else:
            raise NameError(f"{cv_label} is an invalid option - choose one of ['skf','gkf','logo']")
            
        for mood in moods:
            for model in models.keys():
                X, y, groups = self.get_x_and_y(df,mood=mood,additional_features=additional_features)
                clf = models[model]
                scores = cross_val_score(clf, X, y, cv=cv, groups=groups)
                values = [mood,model]+list(scores)+[np.mean(scores)]
                for key, value in zip(res.keys(),values):
                    res[key].append(value)
                    
        return pd.DataFrame(data=res)

    def set_tuned_models(self,dict):
        """
        Sets the tuned models based on the modeling analysis
        """
        self.tuned_models = dict

    def set_tuned_models_bi(self,dict):
        """
        Sets the tuned models based on the modeling analysis for the binary outcome
        """
        self.tuned_models_bi = dict

class Prediction():

    def __init__(self):
        pass

    def get_predictions(self,df,mood,model,probability=False, include_evening=False):
        """
        Gets the predictions for the given mood
        
        """
        X, y, _ = Model.get_x_and_y(df,mood,include_evening=include_evening)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = model.fit(X_train,y_train)
        
        if probability:
            pred = clf.predict_proba(X_test)
        else:
            pred = clf.predict(X_test)
        return y_test, pred

class Evaluation():

    def __init__(self):
        pass

    def get_cm(self,y_true,y_pred,plot=False):
        """
        Returns confusion matrix
        
        Parameters
        ----------
        y_true : list
            the actual values
        y_pred : list
            the predicted values
        plot : boolean
            whether or not to dispaly the confusion matrices
            
        Returns
        -------
        cm : list of lists
            the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if plot:
            _, ax = plt.subplots(figsize=(5,5))
            sns.heatmap(pd.DataFrame(cm),vmin=0,
                        linecolor="black",linewidth=1,cmap="viridis",     
                        square=True,annot=True,fmt='d',ax=ax)
            ax.set_xticklabels(np.arange(len(np.unique(y_true))),fontsize=12,rotation=0)
            ax.set_xlabel("Predicted Value",fontsize=14)
            ax.set_yticklabels(np.arange(len(np.unique(y_true))),fontsize=12,rotation=0)
            ax.set_ylabel("True Value",fontsize=14)
            plt.show()
            plt.close()
        
        return cm

    def get_scoring_metrics(self,df_in,model,binary=False,moods=["content","stress","lonely","sad","energy"],include_evening=False):
        """
        Gets the various scoring metrics
        """
        df = df_in.copy()
        res = {"mood":[],"accuracy":[],"precision":[],"recall":[],"roc_auc":[],"f1":[],"kappa":[],"log-loss":[]}
        for mood in moods:
            y_true, y_pred = Prediction.get_predictions(df,mood,model,include_evening=include_evening)
            _, y_pred_prob = Prediction.get_predictions(df,mood,model,probability=True,include_evening=include_evening)
            res["mood"].append(mood)
            res["accuracy"].append(accuracy_score(y_true,y_pred))
            res["precision"].append(precision_score(y_true,y_pred, average="weighted"))
            res["recall"].append(recall_score(y_true,y_pred, average="weighted"))
            res["f1"].append(f1_score(y_true,y_pred, average="weighted"))
            res["kappa"].append(cohen_kappa_score(y_true,y_pred))
            if binary:
                res["log-loss"].append(log_loss(y_true,y_pred_prob[:, 1]))
                res["roc_auc"].append(roc_auc_score(y_true, y_pred_prob[:, 1], multi_class="ovr", average="weighted"))
            else:
                res["log-loss"].append(log_loss(y_true,y_pred_prob))
                res["roc_auc"].append(roc_auc_score(y_true, y_pred_prob, multi_class="ovr", average="weighted"))
            
        return pd.DataFrame(res)

    def get_report(self,df,model,mood="content"):
        """
        Gets the classification report and prints it
        
        """
        y_true, y_pred = Prediction.get_predictions(df,mood,model)
        print(mood)
        print(classification_report(y_true, y_pred))