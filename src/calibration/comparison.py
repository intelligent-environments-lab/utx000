import sys
sys.path.append('../')

from src.visualization import visualize

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error

# plotting
import matplotlib.pyplot as plt

class Compare:
    
    def __init__(self,param,env1,env2,suffix="wcwh_s21"):
        self.param = param
        self.env1 = env1
        self.env2 = env2
        self.suffix = suffix
        
        self.data = self.import_and_combine(self.param,self.env1,self.env2)

    def import_and_combine(self,param,env1,env2):
        """
        Imports and combines data for the given sensor

        Parameters
        ----------
        param : str
            IAQ parameter to compare
        env1 : str
            name of the first environment
        env2 : str
            name of the second environment

        Returns
        -------
        <df> : DataFrame
            model parameters from both environments
        """
        df1 = pd.read_csv(f"../data/interim/{param}-linear_model_{env1}-{self.suffix}.csv",index_col=0)
        df2 = pd.read_csv(f"../data/interim/{param}-linear_model_{env2}-{self.suffix}.csv",index_col=0)
        
        return df1.merge(df2,left_index=True,right_index=True,suffixes=[f"_{env1}",f"_{env2}"])

class Parameter(Compare):

    def __init__(self, param, env1, env2, suffix):
        super().__init__(param, env1, env2, suffix)
        self.set_diffs()
    
    def set_diffs(self):
        """
        Gets the raw and percent differences between environemnts

        Updates
        -------
        <data> : DataFrame
            Includes raw and percent difference columns
        """
        for x in ["constant","coefficient"]:
            self.data[f"{x}_diff"] = self.data[f"{x}_{self.env1}"] - self.data[f"{x}_{self.env2}"]
            self.data[f"{x}_per_diff"] = self.data[f"{x}_diff"] / self.data[f"{x}_{self.env1}"] * 100
            
    def plot_diffs(self,save=False):
        """
        Plots the percent differences
        """
        df = self.data.copy()
        df.index = df.index.map(str)

        _, ax = plt.subplots(figsize=(16,4))
        ax.scatter(df.index,df["constant_per_diff"],s=50,color="black",zorder=10,label="$x_0$")
        ax.scatter(df.index,df["coefficient_per_diff"],s=60,color="firebrick",marker="s",zorder=1,label="$x_1$")
        ax.axhline(0,lw=2,ls="dashed",color="black")
        # x-axis
        ax.set_xlabel("Device Number",fontsize=16)
        # y-axis
        ax.set_ylabel("Percent Change",fontsize=16)
        # other
        ax.tick_params(labelsize=14)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend(frameon=False,fontsize=14)

        if save:
            plt.savefig(f"../reports/figures/calibration-parameter_difference-{self.param}-{self.suffix}.pdf",bbox_inches="tight")
        plt.show()
        plt.close()

class Output(Compare):

    def __init__(self, param, env1, env2, suffix, ref_env):
        super().__init__(param, env1, env2, suffix)
        self.ref = pd.read_csv(f"../data/interim/{param}-reference-{ref_env}-{self.suffix}.csv")
        self.test_data = pd.read_csv(f"../data/interim/{param}-test_data-{ref_env}-{self.suffix}.csv")

    def get_errors(self, use_average=False,one_minus_two=True):
        """
        Gets the errors/differences between environments

        """
        res = {"beacon":[],"mean_error":[],"min_error":[],"max_error":[]}
        corrected = pd.DataFrame()
        if one_minus_two:
            print(f"Differences Calculated as: {self.env1} - {self.env2}")
        else:
            print(f"Differences Calculated as: {self.env2} - {self.env1}")
        for bb in self.data.index:
            # beacon-specific data
            data_bb = self.test_data[self.test_data["beacon"] == bb]
            data_bb_corrected = pd.DataFrame()
            # model parameters
            if use_average:
                # getting average model parameters
                b1 = np.nanmean(self.data[f"constant_{self.env1}"])
                b2 = np.nanmean(self.data[f"constant_{self.env2}"])
                m1 = np.nanmean(self.data[f"coefficient_{self.env1}"])
                m2 = np.nanmean(self.data[f"coefficient_{self.env2}"])
            else:
                # getting device-specific parameters
                b1 = self.data.loc[bb,f"constant_{self.env1}"]
                b2 = self.data.loc[bb,f"constant_{self.env2}"]
                m1 = self.data.loc[bb,f"coefficient_{self.env1}"]
                m2 = self.data.loc[bb,f"coefficient_{self.env2}"]
            # timeseries data per beacon
            bb_data1 = b1 + m1*data_bb[self.param]
            bb_data2 = b2 + m2*data_bb[self.param]
            #data_bb_corrected["raw"] = data_bb
            #print(data_bb_corrected)
            for col, values in zip(["raw",self.env1,self.env2,"beacon","timestamp"],[data_bb[self.param],bb_data1,bb_data2,bb,data_bb["timestamp"]]):
                data_bb_corrected[col] = values
            
            corrected = corrected.append(data_bb_corrected.set_index("timestamp"))
            # errors
            if one_minus_two:
                errors = bb_data1.values - bb_data2.values
            else:
                errors = bb_data2.values - bb_data1.values
            for key, val in zip(res.keys(),[bb,np.nanmean(errors),np.nanmin(errors),np.nanmax(errors)]):
                res[key].append(val)

        self.errors = pd.DataFrame(res).set_index("beacon")
        self.corrected = corrected

    def plot_errors(self):
        """
        Plots the errors between model outputs
        """
        try:
            res = self.errors.copy()
        except AttributeError:
            print("errors have not been defined - use get_errors method")
            return

        res.sort_values("mean_error",axis=0,inplace=True)
        res.index = res.index.astype("str")

        _, ax = plt.subplots(figsize=(16,5))
        # range lines
        for bb in res.index:
            ax.plot([bb,bb],[res.loc[bb,"min_error"],res.loc[bb,"max_error"]],lw=2,color="black",zorder=1)

        # mean differences
        ax.scatter(res.index,res["mean_error"],color="black",s=50,zorder=2)
        # x-axis
        ax.set_xlabel("Device Number",fontsize=16)
        # y-axis
        ax.set_ylabel(f"Difference in {visualize.get_label(self.param)} ({visualize.get_units(self.param)})",fontsize=16)
        if self.param == "co2":
            ax.set_ylim([-150,500])
        # other
        ax.tick_params(labelsize=14)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

        plt.show()
        plt.close()
    
    def get_reference_limits(self,data=None):
        """
        Gets the minimum and maximum reference given the CGS accuracy
        """
        if isinstance(data,pd.DataFrame):
            pass
        else:
            data = self.ref

        if self.param == "co2":
            low_values = data["concentration"] - (30 + data["concentration"]*0.03)
            high_values = data["concentration"] + (30 + data["concentration"]*0.03)
        elif self.param == "pm2p5_mass":
            low_values = data["concentration"] - 10
            high_values = data["concentration"] + 10
        else:
            print(f"Parameter {self.param} not included in method")
            return data, data

        return low_values, high_values

    def plot_corrected_measurements(self):
        """
        Plots the timeseries of the corrected measurements
        """
        corrected_data = self.corrected.copy()
        # getting experiment time
        data_b1 = corrected_data[corrected_data["beacon"] == 1]
        combined = data_b1.merge(self.ref,left_index=True,right_on="timestamp",how="left")
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined["elapsed_time"] = (combined["timestamp"] - combined["timestamp"].iloc[0]).dt.total_seconds()/60

        _, ax = plt.subplots(figsize=(16,8))
        for bb in corrected_data["beacon"].unique():
            data_bb = corrected_data[corrected_data["beacon"] == bb]
            if bb == 1:
                ax.plot(combined["elapsed_time"],data_bb[self.env1],lw=2,color="orange",label=f"{self.env1.title()} Model",zorder=10)
                ax.plot(combined["elapsed_time"],data_bb[self.env2],lw=2,color="cornflowerblue",label=f"{self.env2.title()} Model",zorder=11)
            else:
                ax.plot(combined["elapsed_time"],data_bb[self.env1],lw=2,color="orange",zorder=10)
                ax.plot(combined["elapsed_time"],data_bb[self.env2],lw=2,color="cornflowerblue",zorder=11)

        # plotting reference band
        low_values, high_values = self.get_reference_limits(data=combined)
        ax.fill_between(combined["elapsed_time"],low_values,high_values,color="black",alpha=0.5,zorder=1,label="Reference")

        # x-axis
        ax.set_xlabel("Experiment Time (mins)",fontsize=16)
        ax.set_xlim([0,120])
        # y-axis
        ax.set_ylabel(f"{visualize.get_label(self.param)} ({visualize.get_units(self.param)})",fontsize=16)
        ax.set_ylim(bottom=0)
        # other
        ax.legend(fontsize=14,frameon=False)
        ax.tick_params(labelsize=14)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        plt.show()
        plt.close()

class IntramodelComparison():
    
    def __init__(self,ieq_param="co2",env="chamber",model_type="linear",study_suffix="wcwh_s21"):
        """
        Initializing function

        Parameters
        ----------
        env : str, default "chamber"
            specifies the environment - one of ["chamber","testhouse"]
        """
        
        self.ieq_param = ieq_param
        self.env = env
        self.model_type = model_type
        self.study_suffix = study_suffix
        
        self.models = {}
        for i in range(3):
            try:
                temp = pd.read_csv(f"../data/interim/{self.ieq_param}-{self.model_type}_model_{self.env}{i+1}-wcwh_s21.csv")
                self.models[i+1] = temp
            except FileNotFoundError as e:
                print(e)
                
    def intra_coeff_comparison(self,save=False,):
        """Compares the coefficients from the two models"""
        params = self.get_coeff_table() # combined parameter table
        if self.model_type == "linear":
            params_to_consider = ["constant","coefficient"]
            params = params[abs(params.sum(axis=1)) != 3] # removing values where the coefficients sum to 3 (default)
        else:
            params_to_consider = ["constant"]
            params = params[abs(params.sum(axis=1)) != 0] # removing values where the constants sum to 0 (default)
        for coeff in params_to_consider:
            h = 0.4  # the height of the bars
            
            
            _, ax = plt.subplots(figsize=(5,8))
            for i, (model_name, color) in enumerate(zip(self.models.keys(),["firebrick","cornflowerblue","seagreen"])):
                y = np.arange(len(params))  # the label locations
                ax.barh(y=y - i * h/(len(self.models)-1), width=params[f"{coeff}{i+1}"], height=h,
                             edgecolor="black", color=color, label=model_name)

            # x-axis
            ax.set_xlabel(f"{coeff.title()} Value",fontsize=16)
            ax.tick_params(axis="x",labelsize=14)

            # y-axis
            ax.set_ylabel('BEVO Beacon Number',fontsize=16)
            ax.set_yticks(y)
            ax.set_yticklabels(params.index,fontsize=14)
            # remaining
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            ax.legend(title="Experiment Number",title_fontsize=14,loc="upper center", bbox_to_anchor=(0.5,-0.1),frameon=False,ncol=3,fontsize=14)

            if save:
                plt.savefig(f"../reports/figures/beacon_summary/calibration_comparison-{coeff}_{self.env}-{self.ieq_param}-{self.study_suffix}.pdf",)

            plt.show()
            plt.close()
            
    def get_coeff_table(self,save=False,latex=False):
        """gets a single dataframe with the model parameters from each experiment"""
        if self.model_type == "linear":
            tab = self.models[1][["beacon","constant","coefficient"]].merge(right=self.models[2][["beacon","constant","coefficient"]],on="beacon",suffixes=("","2")).merge(right=self.models[3][["beacon","constant","coefficient"]],on="beacon",suffixes=("1","3"))
            tab.set_index("beacon",inplace=True)
            tab = tab[["constant1","constant2","constant3","coefficient1","coefficient2","coefficient3"]]
        else:
            tab = pd.DataFrame(data={"beacon":self.models[1]["beacon"],
                         "constant1":self.models[1]["constant"],
                         "constant2":self.models[2]["constant"],
                         "constant3":self.models[3]["constant"]})
            tab.set_index("beacon",inplace=True)

        if save:
            tab.to_csv(f"../data/interim/{self.env}-calibration_model_params-{self.ieq_param}-{self.study_suffix}.csv")

        if latex:
            tab = tab[tab.sum(axis=1) != 3]
            tab.replace([0,1],"--",inplace=True)
            tab = tab.round(decimals=1)
            tab.reset_index(inplace=True)
            print(tab.to_latex(index=False))

        return tab

    def set_averaged_model_params(self):
        """"""
        tab = self.get_coeff_table(save=False)
        tab = tab.replace([0,1], np.nan)
        
        df_to_save = pd.DataFrame()
        for x, val_to_replace in zip(["constant","coefficient"],[0,1]):
            df_to_save[x] = tab[[col for col in tab.columns if col.startswith(x)]].mean(axis=1)
            df_to_save[x].replace(np.nan,val_to_replace,inplace=True)
        
        self.model_params = df_to_save

    def get_beacon_x0(self,bb):
        """gets x0 for specified beacon"""
        try:
            return round(self.model_params.loc[bb,"constant"],2)
        except AttributeError as e:
            print(f"{e} - use set_averaged_model_params to set the model parameters")
            return 0

    def get_beacon_x1(self,bb):
        """gets x1 for specified beacon"""
        try:
            return round(self.model_params.loc[bb,"coefficient"],2)
        except AttributeError as e:
            print(f"{e} - use set_averaged_model_params to set the model parameters")
            return 1

    def set_correction(self,test):
        """Uses model parameters to correct the raw beacon data from a calibration experiment"""
        self.corrected = pd.DataFrame()
        for beacon in test["beacon"].unique():
            test_bb = test[test["beacon"] == beacon]
            test_bb[self.ieq_param] = test[self.ieq_param] * self.get_beacon_x1(beacon) + self.get_beacon_x0(beacon)
            self.corrected = self.corrected.append(test_bb[["timestamp",self.ieq_param,"beacon"]].set_index("timestamp"))

    def set_ref(self,ref):
        """
        Sets the reference
        """
        self.ref=ref

    def show_comprehensive_timeseries(self,ref,r=4,c=5,save=False,show_std=False,scoring_metric="r2",**kwargs):
        """shows a subplot of all the correlation beacons"""
        ref = ref.resample("1T").interpolate()
        fig, axes = plt.subplots(r,c,figsize=(c*4,r*4),sharex=True,sharey=True,gridspec_kw={"wspace":0.1})
        std = np.std(self.corrected[self.ieq_param])
        for bb, ax in zip(self.corrected["beacon"].unique(),axes.flat):
            beacon = self.corrected[self.corrected["beacon"] == bb]
            beacon = beacon.resample("1T").interpolate()
            merged = beacon.merge(right=ref,left_index=True,right_index=True)
            if "resample_rate" in kwargs.keys():
                merged = merged.resample(f"{kwargs['resample_rate']}T").mean()

            t = (merged.index - merged.index[0]).total_seconds()/60
            ax.plot(t,merged[self.ieq_param],color="firebrick",lw=2,zorder=20,label="Corrected")
            ax.plot(t,merged["concentration"],color="black",lw=2,zorder=10, label="Reference")
            if show_std:
                ax.fill_between(t,merged[self.ieq_param]-0.95*std,merged[self.ieq_param]+0.95*std,color="grey",alpha=0.3,zorder=3)
            if "min_val" in kwargs.keys():
                min_val = kwargs["min_val"]
            else:
                min_val = 0

            if "max_val" in kwargs.keys():
                max_val = kwargs["max_val"]
            else:
                max_val = np.nanmax(ref["concentration"])*1.1
            if "zero_line" in kwargs.keys():
                ax.axhline(0,color="gray",lw=2,ls="dashed",alpha=0.5)
                
            ax.set_ylim([min_val,max_val])

            # annotating
            if self.model_type == "linear":
                try:
                    if scoring_metric == "mae":
                        score = mean_absolute_error(merged["concentration"],merged[self.ieq_param])
                        score_label = "MAE"
                    else: # default to r2
                        score = r2_score(merged["concentration"],merged[self.ieq_param])
                        score_label="r$^2$"
                except ValueError:
                    score = 0
                ax.set_title(f"  Device {int(bb)}\n  {score_label} = {round(score,3)}\n  y = {self.get_beacon_x1(bb)}x + {self.get_beacon_x0(bb)}",
                        y=0.85,pad=0,fontsize=13,loc="left",ha="left")
            else:
                ax.set_title(f"  Device {int(bb)}\n  y = x + {self.get_beacon_x0(bb)}",
                        y=0.85,pad=0,fontsize=13,loc="left",ha="left")
                
            ax.set_xticks(np.arange(0,125,30))
            ax.axis('off')

        axes[r-1,0].axis('on')
        for loc in ["top","right"]:
            axes[r-1,0].spines[loc].set_visible(False)
        plt.setp(axes[r-1,0].get_xticklabels(), ha="center", rotation=0, fontsize=16)
        plt.setp(axes[r-1,0].get_yticklabels(), ha="right", rotation=0, fontsize=16)
        axes[r-1,0].legend(loc="upper center",bbox_to_anchor=(0.5,-0.1),ncol=2,frameon=False,fontsize=14)
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel("Experiment Time (min)",fontsize=18)
        plt.ylabel(f"{visualize.get_pollutant_label(self.ieq_param)} ({visualize.get_pollutant_units(self.ieq_param)})",fontsize=18)
        
        # removing any empty axes
        if len(self.corrected["beacon"].unique()) < r*c:
            for i in range(r*c-len(self.corrected["beacon"].unique())):
                axes.flat[-1*(i+1)].axis("off")

        if save:
            plt.savefig(f"../reports/figures/beacon_summary/calibration-{self.ieq_param}-{self.env}-timeseries_comparison-{self.study_suffix}.pdf",bbox_inches="tight")

        plt.show()
        plt.close()

    def get_scores(self):
        """
        Gets the r2 and mae
        """
        res = {"beacon":[],"r2":[],"mae":[]}
        for bb in self.corrected["beacon"].unique():
            beacon = self.corrected[self.corrected["beacon"] == bb]
            beacon = beacon.resample("1T").interpolate()
            merged = beacon.merge(right=self.ref,left_index=True,right_index=True)
            try:
                r2 = r2_score(merged["concentration"],merged[self.ieq_param])
            except ValueError:
                r2 = 0

            try:
                mae = mean_absolute_error(y_true=merged["concentration"],y_pred=merged[self.ieq_param])
            except ValueError:
                mae = np.nan

            for key, val in zip(res.keys(),[bb,r2,mae]):
                res[key].append(val)

        return pd.DataFrame(res).set_index("beacon")

    def save_params(self,**kwargs):
        """Saves the averaged model params
        
        Keyword Arguments:
            - data_path: String specifying the path to (but not including) the "data" dir in the utx000 project
            - study_suffix: String specifying the study suffix to use, defaults to object study
            - env: String to specify the environment, otherwise is left off
        """

        # setting parameters for the save filename
        if "data_path" in kwargs.keys():
            save_dir = kwargs["data_path"]
        else:
            save_dir = "../"
        if "study_suffix" in kwargs.keys():
            study_suffix = kwargs["study_suffix"]
        else:
            study_suffix = self.study_suffix
        if "env" in kwargs.keys():
            env = "_" + kwargs["env"]
        else:
            env = ""
    
        self.model_params.to_csv(f"{save_dir}/data/interim/{self.ieq_param}-{self.model_type}_model{env}-{study_suffix}.csv")
        print("Saved file to:",f"{save_dir}/data/interim/{self.ieq_param}-{self.model_type}_model{env}-{study_suffix}.csv")