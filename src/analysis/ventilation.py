# Ventilation 
# -----------
# Author: Hagen Fritz
# Date: 03/29/21
# Description: 
# -----------
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind, kruskal, boxcox

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# SA
from SALib.sample import saltelli
from SALib.analyze import sobol

from datetime import datetime, timedelta
import math

class calculate():

    def __init__(self, study="utx000", study_suffix="ux_s20", measurement_resolution=120, morning_hours=3, beacons_to_remove=[17,40],data_dir="../../data"):
        """
        Initializating method

        Parameters
        ----------
        study : str, default "utx000"
            study name
        study_suffix : str, default "ux_s20"
            study suffix ID
        measurement_resolution : int, default 120
            interval between beacon measurements in seconds
        morning_hours : int, default 3
            hours to consider after the participant wakes for the decay estimate
        beacons_to_remove : list of int, default [17,40]
            beacons with usable data that shouldn't be used
        data_dir : str, default "../../data"
            path to the "data" directory within the project

        Returns
        -------
        <void>
        """
        self.study = study
        self.suffix = study_suffix
        self.measurement_resolution = measurement_resolution
        self.morning_hours = morning_hours
        self.data_dir = data_dir

        # beacon data
        self.beacon_all = pd.read_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv',index_col="timestamp",parse_dates=True,infer_datetime_format=True)
        self.beacon_nightly = pd.read_csv(f'{self.data_dir}/processed/beacon_by_night-{self.suffix}.csv',
            index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)
        self.beacon_all = self.beacon_all[~self.beacon_all["beacon"].isin(beacons_to_remove)]
        self.beacon_nightly = self.beacon_nightly[~self.beacon_nightly["beacon"].isin(beacons_to_remove)]

        # fitbit data
        self.daily_act = pd.read_csv(f'{self.data_dir}/processed/fitbit-daily-{self.suffix}.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
        #weight_dict = {'beiwe':[],'mass':[]}
        #for pt in self.daily_act['beiwe'].unique():
        #    daily_pt = self.daily_act[self.daily_act['beiwe'] == pt]
        #    weight_dict['beiwe'].append(pt)
        #    weight_dict['mass'].append(np.nanmean(daily_pt['weight'])*0.453592) # convert to kg
        self.sleep_summary = pd.read_csv(f"{self.data_dir}/processed/fitbit-sleep_summary-{self.suffix}.csv",parse_dates=["start_time","end_time","start_date","end_date"],infer_datetime_format=True)
            
        # participant information
        pt_names = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='all')
        pt_names = pt_names[["beiwe","first","last","sex","mass"]]
        pt_ids = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='beacon')
        pt_ids = pt_ids[['redcap','beiwe','beacon','lat','long','volume','roommates']] # keep their address locations
        info = pt_ids.merge(right=pt_names,on='beiwe')
        #mass_info = info.merge(left_on='beiwe',right=pd.DataFrame(weight_dict),right_on='beiwe')
        info['bmr'] = info.apply(lambda row: self.get_BMR(row['sex'],row['mass']),axis=1)
        self.info = info.set_index('beiwe')

    def get_BMR(self, sex, mass):
        '''
        Calculates the BMR based on mass assuming an age range between 18 and 30
        
        Parameters
        ----------
        sex : str
            must be 'Male' or 'Female', otherwise returns 0
        mass : double
            participant mass in kg
        
        Returns
        -------
        <float> : BMR calculated from Persily and De Jong, 2016
        '''
        if sex.lower() == 'male':
            return 0.063*mass + 2.896
        elif sex.lower() == 'female':
            return 0.062*mass + 2.036
        else:
            return 0

    def get_emission_rate(self, BMR, T):
        '''
        Calculates the CO2 emission rate
        
        Paramters
        ---------
        BMR: float
            body-mass-ratio
        T: float
            Temperature in Kelvin
        
        Returns
        -------
        <float> : CO2 emission rate in L/s
        '''
        
        # Assumed constants
        M = 0.95 #METs
        P = 102.5 #kPa
        
        return BMR * M * (T / P) * 0.000179

    def get_volume(self, df, E, n_people=1):
        '''
        Estimates the volume based on the CO2 emission rate and negligible infiltration/exfiltration
        
        Parameters
        ----------
        df : DataFrame 
            indexed by time holding the CO2 and temperature data over an increasing period of CO2
        E : float
            emission rate of CO2 in L/s
        n_people : int, default 1
            specifies number of CO2 emitters
        
        Returns
        -------
        V_si : float
            volume in m3
        V_ip : float
            volume in ft3
        '''
        
        # defining constants
        rho = 1.8 # g/L 
        
        # converting units
        E_gs = E * rho * n_people # L/s to g/s
        df['c'] = self.convert_ppm_to_gm3(df['co2']) # ppm to g/m3
        
        # Calculating
        V_si = (E_gs * (df.index[-1] - df.index[0]).total_seconds()) / (df['c'][-1] - df['c'][0])
        V_ip = V_si / 0.0283168
        return V_si, V_ip

    def estimate_volume_from_co2(self, beacon_data, info, min_window_threshold=8):
        """
        Calculates volumes of the bedrooms for each participant

        Parameters
        ----------
        beacon_data : DataFrame
            IAQ measurements made by the beacons including the participant id, beacon no, co2, and temperature data
        info : DataFrame
            participant demographic info
        min_window_threshold : int, default 8
            minimum length of constinuous measurements

        Returns
        -------
        v_df_averaged : DataFrame
            averaged volume estimates from each participant 
        """
        v_df = pd.DataFrame()
        for pt in beacon_data['beiwe'].unique():
            v_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'starting_co2':[],'co2_delta':[],'ending_co2':[],'R^2':[],'volume_est':[],'volume_gen':[]}
            # getting pt-specific data
            beacon_pt = beacon_data[beacon_data['beiwe'] == pt]
            info_pt = info[info.index == pt]
            # getting periods of co2 increase
            increasing_co2 = self.get_co2_periods(beacon_pt,window=min_window_threshold,change='increase')
            for period in increasing_co2['period'].unique():
                increasing_period_pt = increasing_co2[increasing_co2['period'] == period]
                T = np.nanmean(increasing_period_pt['temperature_c'])
                E = self.get_emission_rate(info_pt.loc[pt,'bmr'],T+273)
                V_gen = info_pt['volume'].values[0]
                V_est = self.get_volume(increasing_period_pt,E)[1]
                # Checking linearity
                Y = increasing_period_pt['co2']
                X = np.arange(0,len(increasing_period_pt)*5,5)
                X = sm.add_constant(X)
                model = sm.OLS(Y,X)
                results = model.fit()
                # adding information to dict
                for key, value_to_add in zip(v_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                        increasing_period_pt.index[0],increasing_period_pt.index[-1],
                                                        increasing_period_pt['co2'][0],
                                                        increasing_period_pt['co2'][-1]-increasing_period_pt['co2'][0],
                                                        increasing_period_pt['co2'][-1],
                                                        results.rsquared,V_est,V_gen]):
                    v_dict[key].append(value_to_add)
                    
            v_df = v_df.append(pd.DataFrame(v_dict))
            
        # Removing bad values
        v_df = v_df[v_df['R^2'] > 0.99]
        v_df = v_df[v_df['volume_est'] < 10000]
        # Averaging
        v_df_averaged = v_df.groupby('beiwe').mean()

        return v_df_averaged

    def get_co2_periods(self, beacon_data, window=30, co2_threshold=10, t_threshold=0.25, time_threshold=120, difference_threshold=30, delta_co2_threshold=60, change='decrease'):
        '''
        Finds and keeps periods of CO2 change or consistency
        
        Parameters
        ----------
        beacon_data : DataFrame
            measured CO2 concentrations
        window : int, default 30
            number of timesteps the increase/decrease period has to last
        co2_threshold : int or float, default 10
            tolerance on the variance in co2 concentration in ppm
        t_threshold : float, default 0.25
            tolerance on temperature variation in C
        time_threshold : float of int, default 120
            maximum time difference, in seconds, between subsequent measurements
        difference_threshold : int or float, default 30 (resolution on SCD30 sensor)
            minimum co2 difference, in ppm, that must occur over the period to be considered - only used when change in ["increase","decrease"]
        delta_co2_threshold : int or float, default 60
            maximum amount of ppm that thee co2 concentration can vary over a periood
        change : str, default "decrease"
            period type to consider - "increase", "decrease", or any other string will specify "constant"
        
        Returns
        -------
        df : DataFrame
            beacon data with only no change/increasing/decreasing periods that satisfy criteria
        '''
        # getting differences
        df = beacon_data.copy()
        df['change'] = df['co2'] - df['co2'].shift(1)
        df['change'] = df['change'].shift(-1)
        
        df['t_change'] = df['temperature_c'] - df['temperature_c'].shift(1)
        df['t_change'] = df['t_change'].shift(-1)
        
        df["time"] = df.index
        df['dtime'] = df["time"] - df["time"].shift(1)
        df['dtime'] = df['dtime'].shift(-1)
        # find periods of increase/decrease and giving them unique labels
        i = 0
        periods = []
        period = 1
        if change == 'decrease':
            while i < len(df):
                while df['change'][i] < 0 and df['t_change'][i] <= 0 and df['dtime'][i].total_seconds() <= time_threshold+2:
                    periods.append(period)
                    i += 1

                periods.append(0)
                period += 1
                i += 1
        elif change == 'increase':
            while i < len(df):
                prior_change = 1000
                while df['change'][i] > 0 and abs(df['t_change'][i]) <= t_threshold and df['dtime'][i].total_seconds() <= time_threshold+2:
                    periods.append(period)
                    i += 1
                    prior_change = df['change'][i]

                periods.append(0)
                period += 1
                i += 1
        else: # constant periods
            while i < len(df):
                i0 = i
                # dummy values
                min_co2 = 600 
                max_co2 = 600
                while abs(df['change'][i]) < co2_threshold and df['t_change'][i] <= 0 and df['dtime'][i].total_seconds() <= time_threshold+2 and max_co2 - min_co2 < delta_co2_threshold:
                    periods.append(period)
                    i += 1
                    min_co2 = np.min(df["co2"][i0:i])
                    max_co2 = np.max(df["co2"][i0:i])

                periods.append(0)
                period += 1
                i += 1
            
        # removing bad periods
        df['period'] = periods
        df = df[df['period'] > 0]
        for period in df['period'].unique():
            temp = df[df['period'] == period]
            # period shorter than window length
            if len(temp) < window:
                df = df[df['period'] != period]
            # difference in concentrations too low - this is doubly checked in the actual ach calculation
            if change in ["increase","decrease"] and abs(temp["co2"].values[-1] - temp["co2"].values[0]) < difference_threshold:
                df = df[df['period'] != period]

        return df

    def convert_ppm_to_gm3(self, concentration, mm=44.0, mv=24.5):
        '''
        Converts the ppm of a gas to g/m3
        
        Parameters
        ----------
        concentration: float
            concentration in ppm
        mm : float, default 44.0 (co2)
            molar mass of the compound
        mv : float, default 24.5 (room temperature)
            molar volume
        
        Returns
        -------
        <float> : concentration in g/m3
        '''
        
        return concentration / 10**6 * mm / mv * 1000

    def convert_gm3_to_ppm(self, concentration, mm=44.0, mv=24.5):
        '''
        Converts the g/m3 of a gas to ppm
        
        Parameters
        ----------
        concentration: float
            concentration in ppm
        mm : float, default 44.0 (co2)
            molar mass of the compound
        mv : float, default 24.5 (room temperature)
            molar volume
        
        Returns
        -------
        <float> : concentration in g/m3
        '''
        
        return concentration * 10**6 / mm * mv / 1000

    def get_param_name(self,param):
        """
        Gets a more complete parameter name
        """
        if param == "v":
            return "Volume"
        elif param == "e":
            return "Emission Rate"
        elif param == "c0":
            return "Background CO$_2$"
        else:
            return ""


    # deprecated
    def get_ventilation_sleep_date(self,estimates,decay=False,verbose=False):
        """
        Gets the wake date associated with the sleep event during which the ventilation was estimated

        Parameters
        ----------
        estimates : DataFrame
            ventilation estimates for each participant
        decay : boolean, default False
            whether the estimates are decay or not
        verbose : boolean, default False
            extra output for debugging purposes

        Returns
        -------
        dates : list of datetime.dates
            dates corresponding to the wake up date from the participant
        """
        dates = []
        for pt in estimates["beiwe"].unique():
            if verbose:
                print(pt)
            est_pt = estimates[estimates["beiwe"] == pt]
            slp_pt = self.sleep_summary[self.sleep_summary["beiwe"] == pt]
            for s_est, e_est in zip(est_pt["start"], est_pt["end"]):
                if verbose:
                    print(f"Starting (estimate):\t{s_est}\nEnding (estimate):\t{e_est}")
                if decay:
                    period_starts = slp_pt["end_time"]
                    period_ends = slp_pt["end_time"]+timedelta(hours=self.morning_hours)
                else:
                    period_starts = slp_pt["start_time"]
                    period_ends = slp_pt["end_time"]
                for s_slp, e_slp in zip(period_starts,period_ends):
                    if verbose:
                        print("\tSleep Periods:")
                        print("\t\t",s_slp)
                        print("\t\t",s_est >= s_slp)
                        print("\t\t",e_slp)
                        print("\t\t",e_est <= e_slp)
                    if s_est >= s_slp and e_est <= e_slp:
                        if verbose:
                            print("\tFound!")
                        dates.append(e_slp.date())
                        break
                        
        return dates

    def summarize_estimates(self,estimates):
        """
        Visually and numerically summarizes the ventilation rates

        Parameters
        ----------
        estimates : DataFrame
            ventilation estimates given in column "ach"

        Returns
        -------
        <void>
        """
        _, ax = plt.subplots(figsize=(12,4))
        sns.kdeplot(x="ach",data=estimates,cut=0,linewidth=2,color="black",ax=ax)
        # x-axis
        ax.set_xlabel("Ventilation Rates",fontsize=16)
        # y-axis
        ax.set_ylabel("")
        # remainder
        ax.tick_params(labelsize=12)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

        plt.show()
        plt.close()

        print(estimates["ach"].describe())

    def diagnose_participant(self,beacon,print_beacon_summary=True,plot_data=True,view_ss=False,view_buildup=False,view_decay=False):
        """
        Runs some diagnostics on participant data and estimated rates

        Parameters
        ----------
        beacon : int
            beacon to consider
        print_beacon_summary : boolean, default True
            output of summary statistics
        plot_data : boolean, default True
            whether to show the time series measurements
        view_ss : boolean, default False
            show the diagnostic plots when calculating ss estimates
        view_buildup : boolean, default False
            show the diagnostic plots when calculating buildup estimates
        view_decay : boolean, default False
            show the diagnostic plots when calculating decay estimates

        Returns
        -------
        data_all : DataFrame
            all beacon data
        data_nightly : DataFrame
            beacon data from sleep periods only
        data_morning : DataFrame
            beacon data after sleep periods
        """
        data_all = self.beacon_all[self.beacon_all["beacon"] == beacon]
        data_nightly = self.beacon_nightly[self.beacon_nightly["beacon"] == beacon]
        try:
            data_morning = self.beacon_morning[self.beacon_morning["beacon"] == beacon]
        except AttributeError:
            # only used decay estimates
            data_morning = pd.DataFrame() # dummy

        # printing some summary statistics on the beacon data
        if print_beacon_summary:
            for measurement_type, df in zip(["Study","Night","Morning"],[data_all,data_nightly,data_morning]):
                try:
                    print(f"{measurement_type} Measurements")
                    print(f"\tMin:\t{np.nanmin(df['co2'])}\n\tMean:\t{np.nanmean(df['co2'])}\n\tMax:\t{np.nanmax(df['co2'])}")
                except KeyError:
                    # no data in data_morning
                    pass

        # plotting time series
        if plot_data:
            if len(data_morning) > 1:
                _, axes = plt.subplots(3,1,figsize=(16,12),sharex=True,gridspec_kw={"hspace":0,"wspace":0})
            else:
                _, axes = plt.subplots(2,1,figsize=(16,8),sharex=True)

            for df, ax in zip([data_all,data_nightly,data_morning],axes):
                # scatter
                ax.scatter(df.index,df["co2"],s=5,color="black",zorder=100)
                # percentile lines
                for p in [10,25,50,75,90]:
                    ax.axhline(np.nanpercentile(df["co2"],p),color="firebrick",linestyle="dashed",zorder=p)
                # x-axis
                ax.set_xlim([data_all.index[0],data_all.index[-1]])
                # y-axis
                ax.set_ylim([np.nanmin(data_all["co2"]),np.nanmax(data_all["co2"])])

            plt.show()
            plt.close()

        if view_ss:
            ss_calc = steady_state(data_dir="../data")
            _ = ss_calc.ventilation_ss(data_nightly,self.info,constant_c0=False,c0_percentile=0,
                min_co2_threshold=None,co2_threshold_percentile=50,plot=True)
        if view_buildup:
            dynamic_calc = dynamic(data_dir="../data")
            _ = dynamic_calc.ventilation_buildup(data_nightly, self.info, delta_co2_threshold=120,decreasing_increase=False,decreasing_increase_rate=5,plot=True)
        if view_decay:
            dynamic_calc = dynamic(data_dir="../data")
            _ = dynamic_calc.ventilation_decay(data_morning, self.info, constant_c0=False, c0_percentile=0, delta_co2_threshold=120, plot=True)

        return data_all, data_nightly, data_morning

    def inspect_c0(self,percentile=0,by_id="beacon"):
        """
        Inspects the c0 values

        Parameters
        ----------
        percentile : float, default 0 (min)
            value between [0,1] to compute the percentile
        by_id : str, default "beacon"
            participant ID type

        Returns
        -------
        <c0> : DataFrame
            C0 values for each participant for all and nightly measurements
        """
        temp_all = self.beacon_all[[by_id,"co2"]].groupby(by_id).quantile(percentile)
        temp_nightly = self.beacon_nightly[[by_id,"co2"]].groupby(by_id).quantile(percentile)
        return temp_all.merge(temp_nightly,left_index=True,right_index=True,suffixes=["_all","_nightly"])

    def inspect_pt_c0(self, pt, by_id="beacon", calc=None):
        """
        Inspects participant C0 measurements

        Parameters
        ----------
        pt : int or str
            participant ID
        by_id : str, default "beacon"
            participant ID type
        calc : str, default None
            ventilation calculation method to run if desired

        Returns
        -------
        res : DataFrame
            results from the calculation
            if no calc is specified, returns None
        """
        bb_pt_all = self.beacon_all[self.beacon_all[by_id] == pt]
        bb_pt_nightly = self.beacon_nightly[self.beacon_nightly[by_id] == pt]
        _, ax = plt.subplots(figsize=(20,5))
        ax.plot(bb_pt_all.index,bb_pt_all["co2"],color="black",label="All")
        ax.scatter(bb_pt_nightly.index,bb_pt_nightly["co2"],s=5,color="firebrick",label="Nightly")
        # x-axis
        ax.set_xlim([bb_pt_all.index[0],bb_pt_all.index[-1]])
        # y-axis
        ax.set_ylabel("Concentration (ppm)",fontsize=14)
        # remainder
        ax.tick_params(labelsize=12)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

        ax.legend(frameon=False,fontsize=14)

        plt.show()
        plt.close()

        _, axes = plt.subplots(1,2,figsize=(20,5),sharey=True)
        for data, color, ax in zip([bb_pt_all,bb_pt_nightly],["black","firebrick"],axes):
            sns.kdeplot("co2",data=data,lw=3,color=color,ax=ax)
            # x-axis
            ax.set_xlabel("Concentration (ppm)",fontsize=14)
            ax.set_xlim([np.nanmin(bb_pt_all["co2"]),np.nanmax(bb_pt_all["co2"])])
            # y-axis
            ax.set_ylabel("Density",fontsize=14)
            # remainder
            ax.tick_params(labelsize=12)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)

        plt.show()
        plt.close()

        if calc == "ss":
            ss = steady_state(data_dir="../data")
            res_constant = ss.ventilation_ss(bb_pt_nightly, self.info, constant_c0=True)
            res_constant["method"] = "Constant C0"
            res_percentile = ss.ventilation_ss(bb_pt_nightly, self.info, constant_c0=False)
            res_percentile["method"] = "Percentile C0"
            res_pt = pd.concat([res_constant,res_percentile])
        else:
            res_pt = None

        return res_pt

class steady_state(calculate):

    def get_ach_from_constant_co2(self, E, V, C, C0=450.0, p=1.0):
        '''
        Calculates the air exchange rate for constant CO2 events
        
        Parameters
        ----------
        E : float
            emission rate in L/s
        V : float
            volume in ft3
        C : float 
            room co2 concetration in ppm
        C0 : float, default 450.0
            outdoor co2 concentration in ppm
        p: float, default 1.0
            penetration factor
        
        Returns
        -------
        <float> : ach in 1/h
        '''
        # defining constants
        rho = 1.8 # g/L 
        
        # converting units
        E_gs = E * rho # L/s to g/s
        V_m3 = V * 0.0283168 # ft3 to m3
        C_gm3 = self.convert_ppm_to_gm3(C) # ppm to g/m3
        C0_gm3 = self.convert_ppm_to_gm3(C0) # ppm to g/m3
        
        return E_gs / (V_m3 * (C_gm3 - p*C0_gm3)) * 3600

    def ventilation_ss(self, beacon_data, info, constant_c0=False, c0_percentile=0,
        data_length_threshold=90, min_window_threshold=30, min_co2_threshold=None, co2_threshold_percentile=50, min_time_threshold=120, 
        plot=False, save_plot=False,**kwargs):
        """
        Gets all possible ventilation rates from the steady-state assumption
        
        Parameters
        ---------
        beacon_data : DataFrame
            IAQ measurements made by the beacons including the participant id, beacon no, co2, and temperature data
        info : DataFrame
            participant demographic info
        constant_c0 : boolean, default True
            whether to use constant background concentration or participant-based if set to False
        c0_percentile : int or float, default 1
            percentile of participant-measured co2 to use as background baseline if not using constant background
        data_length_threshold : int, default 90 (3 hours of 2-minute resolution data)
            minimum number of nightly datapoints that must be available i.e. participant needs to sleep a certain length of time
        min_window_threshold : int, default 12
            minimum length of continuous measurements
        min_co2_threshold : int or float, default None
            minimum average co2 concentration that must be measured during the period.
            If None, a percentile value based on all the participants measurements is used. 
        co2_threshold_percentile : int or float, default 50
            percentile to use for participant-based co2 threshold
        min_time_threshold : int or float, default 120
            the minimum time (in seconds) between measurements
        plot : boolean, default False
            plot diagnostic plots of each identified period
        save_plot : boolean, default False
            whether or not save diagnostic plots
        
        Returns
        -------
        ventilation_df : DataFrame
            ventilation rates estimated for each steady-state period
        """
        ventilation_df = pd.DataFrame()
        self.n_ss_periods = 0
        for pt in beacon_data['beiwe'].unique(): # cycling through each of the participants
            # setting up the dictionary to add pt values to
            pt_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'co2_mean':[],'co2_delta':[],'t_mean':[],'t_delta':[],'e':[],'c0':[],"v":[],'ach':[]}
            
            # pt-specific data
            # ----------------
            beacon_pt = beacon_data[beacon_data["beiwe"] == pt]
            beacon_all = self.beacon_all[self.beacon_all["beiwe"] == pt]
            info_pt = info[info.index == pt]
            ## C0
            if constant_c0:
                C0 = 450
            else: # pt-based from all available data
                C0 = np.nanpercentile(beacon_all["co2"],c0_percentile)
            
            ## CO2 threshold
            if min_co2_threshold:
                min_co2_value = min_co2_threshold
            else:
                min_co2_value = np.nanpercentile(beacon_pt["co2"],co2_threshold_percentile)

            # looping through pt-specific sleep events registered by FB
            for start, end in zip(beacon_pt['start_time'].unique(),beacon_pt['end_time'].unique()): 
                beacon_pt_night = beacon_pt[start:end] # masking for iaq data during sleep
                if len(beacon_pt_night) > data_length_threshold: # must have slept for a certain length
                    # getting constant periods per sleep event
                    constant_periods = self.get_co2_periods(beacon_pt_night[['co2','temperature_c','rh']],window=min_window_threshold,time_threshold=min_time_threshold,change='constant')
                    if len(constant_periods) > 0:
                        # summarizing the constant period(s)
                        self.n_ss_periods += len(constant_periods['period'].unique())
                        for period in constant_periods['period'].unique():
                            constant_by_period = constant_periods[constant_periods['period'] == period]
                            C = np.nanmean(constant_by_period['co2'])
                            if C >= min_co2_value: # must have measured co2 over a certain threshold to be considered
                                # calculating relevant parameters per period
                                dC = np.nanmean(constant_by_period['change'])
                                T = np.nanmean(constant_by_period['temperature_c'])
                                dT = np.nanmean(constant_by_period['t_change'])
                                if "v" in kwargs.keys():
                                    V = kwargs["v"]
                                else:
                                    V = info_pt['volume'].values[0]
                                if "e" in kwargs.keys():
                                    E = kwargs["e"]
                                else:
                                    E = self.get_emission_rate(self.info.loc[pt,'bmr'],T+273)
                                # calculating ventilation rate
                                ACH = self.get_ach_from_constant_co2(E,V,C,C0)
                                # appending data to pt-specific dictionary
                                for k, v in zip(["beiwe","beacon","start","end","co2_mean","co2_delta","t_mean","t_delta","e","c0","v","ach"],
                                                [pt,info_pt['beacon'].values[0],start,end,C,dC,T,dT,E,C0,V,ACH]):
                                    pt_dict[k].append(v)

                                # diagnostics
                                if plot:
                                    self.plot_constant_co2_period(constant_by_period,ACH,pt,save=save_plot)

            ventilation_df = ventilation_df.append(pd.DataFrame(pt_dict))
            ventilation_df = ventilation_df.groupby(["start","end","beiwe"]).mean().reset_index()
            
        return ventilation_df

    def plot_constant_co2_period(self, df, ach, pt="", save=False):
        """
        plots the co2 and temperature from a constant co2 period
        
        Parameters
        ----------
        df : DataFrame
            data
        ach : float
            air exchange rate
        pt : str, default empty string
            participant id
        save : boolean, default False
            whether or not to save the figure
            
        Returns
        -------
        <void>
        """
        _, ax = plt.subplots(figsize=(8,6))
        # co2 concentraiton (left axis)
        ax.plot(df.index,df["co2"],color="seagreen")
        ax.set_ylim([400,2000])
        ax.set_ylabel("CO$_2$ (ppm)",fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14,ha="left",rotation=-45)
        
        ax2 = ax.twinx()
        # temperature (right axis)
        ax2.plot(df.index,df["temperature_c"],color="cornflowerblue")
        ax2.spines['right'].set_color('cornflowerblue')
        ax2.set_ylim([20,30])
        plt.yticks(fontsize=14)
        ax2.set_ylabel("Temperature ($^\circ$C)",fontsize=16,color="cornflowerblue")
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(fontsize=14,ha="left",rotation=-45)

        # annotating
        period = df['period'][0]
        ax.set_title(f"ID: {pt} - Period: {period} - ACH: {round(ach,2)} - Date: {datetime.strftime(df.index[-1],'%Y-%m-%d')}")
        
        if save:
            plt.savefig(f"../reports/figures/beacon_summary/ventilation_estimates/method_0-{pt}-{period}.pdf",bbox_inches="tight")
            
        plt.show()
        plt.close()
    
    def get_occupancy_detection_rates(self, occ_obj, percents=[0.7,0.8,0.9], plot=False, save=False, save_dir="../reports/figures/"):
        """
        Plots ventilation rates from various confidence thresholds
        
        Parameters
        ----------
        occ_obj : instance of class Occupancy Detection Classify
            occupancy detection model
        percents : list of float, default [0.7,0.8,0.9]
            list of confidence thresholds
        plot : boolean, default False
            whether to plot the estimates or not
        save : boolean, default False
            whether to save the figure or not
        save_dir : str, default "../reports/figures/"
            path to save the figure

        Returns
        -------
        estimates_ss_df : DataFrame
            estimated ventilation rates by threshold
        """

        estimates_ss_agg = {}
        for p in percents:
            occupied_data, _ = occ_obj.get_occupied_iaq_data(confidence_threshold=p)#,confidence_limit=lim)
            estimates_occupied = self.ventilation_ss(occupied_data,self.info,constant_c0=False,c0_percentile=0,min_co2_threshold=None,
                                                        data_length_threshold=6,min_window_threshold=6,min_time_threshold=600,co2_threshold_percentile=50)
            estimates_ss_agg[f"{int(p*100)}"] = estimates_occupied

        estimates_ss_df = pd.DataFrame()
        for key in estimates_ss_agg.keys():
            # adding columns to group on
            estimates_ss_agg[key]["method"] = key
            estimates_ss_df = estimates_ss_df.append(estimates_ss_agg[key])
        
        fig, axes = plt.subplots(len(percents),1,figsize=(10,4*len(percents)),sharex=True)
        if plot:
            for p, ax in zip(percents,axes):
                data_percent = estimates_ss_df[estimates_ss_df["method"] == f"{int(p*100)}"]

                device_no = []
                for bb in data_percent["beacon"]:
                    if bb < 10:
                        device_no.append("0"+str(int(bb)))
                    else:
                        device_no.append(str(int(bb)))

                data_percent["device"] = device_no
                data_percent.sort_values("device",inplace=True)
                sns.stripplot(x="device",y="ach",color="black",edgecolor="black",linewidth=1,size=8,jitter=0.25,alpha=0.33,data=data_percent,ax=ax)
                # x-axis
                ax.set_xlabel("")
                #y-axis
                ax.set_yticks([0,1,2,3])
                ax.set_ylabel("")
                # other
                ax.tick_params(labelsize=16)
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
                ax.set_title(f"{int(p*100)}% Confidence",fontsize=18)
                
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.ylabel("Ventilation Rate (h$^{-1}$)",fontsize=22)
            ax.set_xlabel("ID of Participants with Sufficient Data",fontsize=22)
            
            if save:
                plt.savefig(f"{save_dir}/beacon_summary/ventilation_rates-occupancy_detection-strip-ux_s20.pdf",bbox_inches="tight")
                
            plt.show()
            plt.close()

        return estimates_ss_df

class base_case(steady_state):

    def get_base_data(self,data=None,start_time_hour=1,end_time_hour=4):
        """
        Gets the base case data (weekday between early morning times)

        Parameters
        ----------
        data : DataFrame, default None
            data to filter
        start_time_hour : int, default 1
            intial hour to consider on 24-hour clock
        end_time_hour : int, default 4
            final hour to consider on a 24-hour clock

        Returns
        -------
        df : DataFrame
            filetered dataframe
        """
        if data is None:
            try:
                df = self.beacon_all.copy()
            except AttributeError:
                print("No beacon_all attribute")
                return pd.DataFrame()
        else:
            df = data.copy

        # filtering for weekday
        df["dow"] = df.index.weekday
        df = df[~df["dow"].isin([5,6])]
        # filtering for time range
        df = df[df.index.hour.isin(range(start_time_hour,end_time_hour))]
    
        return df

    def ventilation_ss(self, beacon_data, info, constant_c0=False, c0_percentile=0,
        min_window_threshold=30, min_co2_threshold=None, co2_threshold_percentile=50, min_time_threshold=120, 
        plot=False, save_plot=False,**kwargs):
        """
        Gets all possible ventilation rates from the steady-state assumption
        
        Parameters
        ---------
        beacon_data : DataFrame
            IAQ measurements made by the beacons including the participant id, beacon no, co2, and temperature data
        info : DataFrame
            participant demographic info
        constant_c0 : boolean, default True
            whether to use constant background concentration or participant-based if set to False
        c0_percentile : int or float, default 1
            percentile of participant-measured co2 to use as background baseline if not using constant background
        min_window_threshold : int, default 12
            minimum length of continuous measurements
        min_co2_threshold : int or float, default None
            minimum average co2 concentration that must be measured during the period.
            If None, a percentile value based on all the participants measurements is used. 
        co2_threshold_percentile : int or float, default 50
            percentile to use for participant-based co2 threshold
        min_time_threshold : int or float, default 120
            the minimum time (in seconds) between measurements
        plot : boolean, default False
            plot diagnostic plots of each identified period
        save_plot : boolean, default False
            whether or not save diagnostic plots
        
        Returns
        -------
        ventilation_df : DataFrame
            ventilation rates estimated for each steady-state period
        """
        ventilation_df = pd.DataFrame()
        self.n_ss_periods = 0
        for pt in beacon_data['beiwe'].unique(): # cycling through each of the participants
            # setting up the dictionary to add pt values to
            pt_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'co2_mean':[],'co2_delta':[],'t_mean':[],'t_delta':[],'e':[],'c0':[],"v":[],'ach':[]}
            
            # pt-specific data
            # ----------------
            beacon_pt = beacon_data[beacon_data["beiwe"] == pt]
            beacon_all = self.beacon_all[self.beacon_all["beiwe"] == pt]
            info_pt = info[info.index == pt]
            ## C0
            if constant_c0:
                C0 = 450
            else: # pt-based from all available data
                C0 = np.nanpercentile(beacon_all["co2"],c0_percentile)
            
            ## CO2 threshold
            if min_co2_threshold:
                min_co2_value = min_co2_threshold
            else:
                min_co2_value = np.nanpercentile(beacon_pt["co2"],co2_threshold_percentile)

            constant_periods = self.get_co2_periods(beacon_pt[['co2','temperature_c','rh']],window=min_window_threshold,time_threshold=min_time_threshold,change='constant')
            if len(constant_periods) > 0:
                # summarizing the constant period(s)
                self.n_ss_periods += len(constant_periods['period'].unique())
                for period in constant_periods['period'].unique():
                    constant_by_period = constant_periods[constant_periods['period'] == period]
                    C = np.nanmean(constant_by_period['co2'])
                    if C >= min_co2_value: # must have measured co2 over a certain threshold to be considered
                        # calculating relevant parameters per period
                        dC = np.nanmean(constant_by_period['change'])
                        T = np.nanmean(constant_by_period['temperature_c'])
                        dT = np.nanmean(constant_by_period['t_change'])
                        if "v" in kwargs.keys():
                            V = kwargs["v"]
                        else:
                            V = info_pt['volume'].values[0]
                        if "e" in kwargs.keys():
                            E = kwargs["e"]
                        else:
                            E = self.get_emission_rate(self.info.loc[pt,'bmr'],T+273)
                        # calculating ventilation rate
                        ACH = self.get_ach_from_constant_co2(E,V,C,C0)
                        # appending data to pt-specific dictionary
                        for k, v in zip(["beiwe","beacon","start","end","co2_mean","co2_delta","t_mean","t_delta","e","c0","v","ach"],
                                        [pt,info_pt['beacon'].values[0],constant_by_period.index[0],constant_by_period.index[-1],C,dC,T,dT,E,C0,V,ACH]):
                            pt_dict[k].append(v)

                        # diagnostics
                        if plot:
                            self.plot_constant_co2_period(constant_by_period,ACH,pt,save=save_plot)

            ventilation_df = ventilation_df.append(pd.DataFrame(pt_dict))
            ventilation_df = ventilation_df.groupby(["start","end","beiwe"]).mean().reset_index()
            
        return ventilation_df

class dynamic(calculate):

    def get_morning_beacon_data(self, night_df, all_df, num_hours=3):
        '''
        Grabs beacon data for hours after the participant has woken up.
        
        Parameters
        ----------
        night_df : DataFrame 
            nightly measured beacon values
        all_df : DataFrame
            all measured beacon values
        num_hours : int or float, default 3
            number of hours after waking up to consider
        
        Returns
        -------
        df : DataFrame
            beacon measurements for the "morning" after
        '''
        df = pd.DataFrame()
        for pt in night_df['beiwe'].unique():
            # pt-specific data
            night_pt = night_df[night_df['beiwe'] == pt]
            beacon_pt = all_df[all_df['beiwe'] == pt]
            # getting measurements after wake periods
            for wake_time in night_pt['end_time'].unique():
                temp = beacon_pt[wake_time:pd.to_datetime(wake_time)+timedelta(hours=num_hours)]
                temp['start_time'] = wake_time
                df = df.append(temp)
        
        return df

    def set_beacon_morning(self, df):
        """
        Sets the beacon data for the morning period
        """
        self.beacon_morning = df[['beiwe','beacon','co2','temperature_c','rh','start_time']]

    def get_ach_from_dynamic_co2(self, df, E, V, C0=450.0, p=1.0, measurement_resolution=120, error_metric="rmse",
        plot=False, pt="", period="", method="", save=False):
        '''
        Calculates the ACH based on a dynamic solution to the mass balance equation
        
        Parameters
        ----------
        df : DataFrame 
            indexed by time with CO2 column for CO2 measurements in ppm
        E : float 
            emission rate in L/s
        V : float
            volume in ft3
        C0 : float, default 450.0
            outdoor co2 concentration in ppm
        p : float, default 1.0
            penetration factor
        measurement_resolution : int, default 120
            interval between beacon measurements in seconds
        error_metric : str, default "rmse"
            how to evaluate the estimate
        plot : boolean, default False
            whether to plot the diagnostic decay periods
        pt : str, default ""
            participant beiwe id
        period : str, default ""
            period number (used for diagnostic plot)
        method : str, default ""
            estimation method name
        save : boolean, default False
            whether or not to save the diagnostic plots
        
        Returns
        -------
        ach : float
            air exchange rate in 1/h
        min_rmse : float

        C_to_plot : 

        '''
        # defining constants
        rho = 1.8 # g/L 
        
        # converting units
        E_gh = E * rho *3600 # L/s to g/h
        V_m3 = V * 0.0283168 # ft3 to m3
        df['c'] = self.convert_ppm_to_gm3(df['co2']) # ppm to g/m3
        C0_gm3 = self.convert_ppm_to_gm3(C0) # ppm to g/m3
        
        # initial values
        C_t0 = df['c'][0]
        min_error = math.inf
        ach = -1
        C_to_plot = df['c'].values # for comparison
        # looping through possible ach values
        for ell in np.arange(0,10.001,0.001):
            Cs = []
            for i in range(len(df)):
                t = i*measurement_resolution/3600
                Cs.append(C_t0 * math.exp(-ell*t) + (p*C0_gm3 + E_gh/(V_m3*ell))*(1 - math.exp(-ell*t)))
                
            # calculating error metric(s)
            error = 0
            if error_metric == "rmse":
                for C_est, C_meas in zip(Cs,df['c']):
                    error += (C_est-C_meas)**2
                error = math.sqrt(error/len(Cs))
            else:
                for C_est, C_meas in zip(Cs,df['c']):
                    error += abs(C_est-C_meas)
                error = error/len(Cs)

            # saving best result
            if error < min_error:
                min_error = error
                ach = ell
                C_to_plot = Cs

        if ell in [0,0.001,10]:
            ell = np.nan
                
        # plotting to compare results
        if plot:
            measured_co2 = self.convert_gm3_to_ppm(df['c'])
            calculated_co2 = list(map(self.convert_gm3_to_ppm,C_to_plot))
            _, ax = plt.subplots(figsize=(8,6))
            # concentration axis
            ax.plot(df.index,measured_co2,color='seagreen',label='Measured')
            ax.plot(df.index,calculated_co2,color='firebrick',label=f'ACH={round(ach,2)}; {error_metric.upper()}={round(error,3)}')

            for i in range(len(Cs)):
                ax.annotate(str(round(measured_co2.values[i],1)),(df.index[i],measured_co2.values[i]),ha="left",fontsize=12)
                ax.annotate(str(round(calculated_co2[i],1)),(df.index[i],calculated_co2[i]),ha="right",fontsize=12)
                
            ax.set_ylabel("CO$_2$ (ppm)",fontsize=16)
            plt.yticks(fontsize=14)
            ax.legend(fontsize=14)
            plt.xticks(fontsize=14,ha="left",rotation=-45)

            ax2 = ax.twinx()
            # temperature axis
            ax2.plot(df.index,df['temperature_c'],color='cornflowerblue',label='Temperature')
            ax2.spines['right'].set_color('cornflowerblue')
            ax2.set_ylim([20,30])
            plt.yticks(fontsize=14)
            ax2.set_ylabel("Temperature ($^\circ$C)",fontsize=16,color="cornflowerblue")
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

            ax.set_title(f"ID: {pt} - Period: {period}")
            if save:
                plt.savefig(f"../reports/figures/beacon_summary/ventilation_estimates/method_{method}-{pt}-{period}.pdf",bbox_inches="tight")
            plt.show()
            plt.close()
            
        return ach, min_error, C_to_plot

    # deprecated
    def ventilation_decay_no_occupant_no_penetration(self, beacon_data, info, min_window_threshold=30, min_co2_threshold=600, plot=False, save_plot=False, **kwargs):
        """
        Ventilation estimate based on simple decay equation i.e. no indoor or outdoor sources

        Parameters
        ----------
        beacon_data : DataFrame
            contains beacon data from the morning after participants wake
        info : DataFrame
            participant demographic info
        min_window_threshold : int, default 12
            minimum length of constinuous measurements
        min_co2_threshold : int or float, default 600
            minimum nightly average co2 concentration that must be measured
        plot : boolean, default False
            plot diagnostic plots of each identified period
        save_plot : boolean, default False
            save diagnostic plots 

        Returns
        -------
        decay_df : DataFrame
            ventilation estimates for each decay period for each participant from the simplified model
        """
        decay_df = pd.DataFrame()
        for pt in beacon_data['beiwe'].unique():
            decay_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'ending_co2_meas':[],'ending_co2_calculated':[],'rmse':[],'ach':[]}
            # getting pt-specific data
            beacon_co2_pt = beacon_data[beacon_data['beiwe'] == pt]
            info_pt = info[info.index == pt]
            # getting 
            decreasing_co2_ac_pt = self.get_co2_periods(beacon_co2_pt,window=min_window_threshold,change='decrease')
            for period in decreasing_co2_ac_pt['period'].unique():
                decreasing_period_ac_pt = decreasing_co2_ac_pt[decreasing_co2_ac_pt['period'] == period]
                if np.nanmin(decreasing_period_ac_pt['co2']) >= min_co2_threshold:
                    V = info_pt['volume'].values[0]
                    # assumptions for simplified model
                    E = 0
                    p = 0
                    # estimating ventilation rates
                    ach, ss, C_est = self.get_ach_from_dynamic_co2(decreasing_period_ac_pt,E,V,p,plot=plot,pt=pt,period=period,method=1,save=save_plot)
                    # adding information to dict
                    for key, value_to_add in zip(decay_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                            decreasing_period_ac_pt.index[0],decreasing_period_ac_pt.index[-1],
                                                            decreasing_period_ac_pt['c'][-1],C_est[-1],
                                                            ss,ach]):
                        decay_dict[key].append(value_to_add)
                    
            decay_df = decay_df.append(pd.DataFrame(decay_dict))

        return decay_df

    def ventilation_decay(self, beacon_data, info, 
        constant_c0=False, c0_percentile=0, min_window_threshold=30, min_co2_threshold=600, delta_co2_threshold=None,
        plot=False, save_plot=False,verbose=False,**kwargs):
        """
        Ventilation estimate based on no indoor sources

        Parameters
        ----------
        beacon_data : DataFrame
            contains beacon data from the morning after participants wake
        info : DataFrame
            participant demographic info
        constant_c0 : boolean, default True
            whether to use constant background concentration or participant-based if set to False
        c0_percentile : int or float, default 0
            percentile of participant-measured co2 to use as background baseline if not using constant background
        min_window_threshold : int, default 30
            minimum length of constinuous measurements
        min_co2_threshold : int or float, default 600
            minimum nightly average co2 concentration that must be measured
        delta_co2_threshold : int or float, default None
            minimum change in co2 during periods to consider
        plot : boolean, default False
            plot diagnostic plots of each identified period
        save_plot : boolean, default False
            save diagnostic plots 

        Returns
        -------
        decay_df : DataFrame
            ventilation estimates for each decay period for each participant from the simplified model
        """
        decay_df = pd.DataFrame()
        for pt in beacon_data['beiwe'].unique():
            decay_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'ending_co2_meas':[],'ending_co2_calculated':[],'rmse':[],'ach':[]}
            # getting pt-specific data
            beacon_co2_pt = beacon_data[beacon_data['beiwe'] == pt]
            info_pt = info[info.index == pt]
            if constant_c0:
                C0 = 450
            else: # pt-based
                C0 = np.nanpercentile(self.beacon_all[self.beacon_all["beiwe"] == pt]["co2"],c0_percentile)
            if "v" in kwargs.keys():
                V = kwargs["v"]
            else:
                V = info_pt['volume'].values[0]
            # getting periods with decreasing co2 and T
            decreasing_co2_ac_pt = self.get_co2_periods(beacon_co2_pt,window=min_window_threshold,change='decrease')
            n_periods = len(decreasing_co2_ac_pt['period'].unique())
            if verbose:
                print("Number of decreasing periods:",n_periods)
            for i, period in enumerate(decreasing_co2_ac_pt['period'].unique()):
                decreasing_period_ac_pt = decreasing_co2_ac_pt[decreasing_co2_ac_pt['period'] == period]
                # sorting out the difference threshold
                if delta_co2_threshold:
                    delta_co2_value = delta_co2_threshold
                else:
                    delta_co2_value = np.nanmax(decreasing_period_ac_pt['co2'])/2
                # min and max calculations
                min_co2_period = np.nanmin(decreasing_period_ac_pt['co2'])
                max_co2_period = np.nanmax(decreasing_period_ac_pt['co2'])
                # calculating ventilation rate if conditions are met
                if min_co2_period >= min_co2_threshold and (max_co2_period - min_co2_period) >= delta_co2_value:
                    # calculating
                    if verbose:
                        print(f"Calculating for period {i}/{n_periods}...",end=" ")
                    ach, ss, C_est = self.get_ach_from_dynamic_co2(decreasing_period_ac_pt,E=0,V=V,C0=C0,plot=plot,pt=pt,period=period,method="decay",save=save_plot)
                    if verbose:
                        print("\tACH:",ach)
                    # adding information to dict
                    for key, value_to_add in zip(decay_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                            decreasing_period_ac_pt.index[0],decreasing_period_ac_pt.index[-1],
                                                            decreasing_period_ac_pt['c'][-1],C_est[-1],
                                                            ss,ach]):
                        decay_dict[key].append(value_to_add)
                    
            decay_df = decay_df.append(pd.DataFrame(decay_dict))

        return decay_df

    def get_decreasing_increase_subperiod(self,period_df,resample_rate=2):
        """
        Gets only data from increasing periods where the difference between subsequent CO2 measurements is decreasing on average

        Parameters
        ----------
        period_df : DataFrame
            processed data from increasing CO2 period
        resample_rate : int, default 2
            downsample rate to smooth data
            default of 2 is the minimum downsample rate and corresponds to the original downsample rate (at least for older data)

        Returns
        -------
        subperiod_df : DataFrame
            data from increasing period where differences between subsequent CO2 measurements are decreasing on average
        """
        period_df_resampled = period_df.resample(f"{resample_rate}T").mean()
        period_df_resampled["change_in_change"] = period_df_resampled["change"] - period_df_resampled["change"].shift(1)          

        decreasing_increase = pd.DataFrame()
        for i in range(len(period_df_resampled)-1):
            if period_df_resampled["change_in_change"][(i+1)*-1] < 0:
                decreasing_increase = decreasing_increase.append(period_df_resampled.iloc[(i+1)*-1,:])
            else:
                decreasing_increase.sort_index(inplace=True)
                break
        
        #print(decreasing_increase.head())
        try:
            subperiod_df = period_df[decreasing_increase.index[0]:]
        except IndexError:
            #print(period_df_resampled["change_in_change"])
            subperiod_df = period_df

        return subperiod_df

    # version 1 used in favor over this method
    def ventilation_buildup(self, beacon_data, info, 
        constant_c0=False, c0_percentile=0, min_window_threshold=30, min_co2_threshold=600, delta_co2_threshold=120, truncate_threshold=0,
        decreasing_increase=False,decreasing_increase_rate=2,plot=False, save_plot=False,**kwargs):
        """
        Ventilation estimate based on initial occupancy. This version truncates the first series of measurements until the difference
        between subsequent CO2 measurements is greater than the truncate_threshold

        Parameters
        ----------
        beacon_data : DataFrame
            IAQ measurements made by the beacons including the participant id, beacon no, co2, and temperature data
        info : DataFrame
            participant demographic info
        constant_c0 : boolean, default False
            whether to use constant background concentration or participant-based if set to False
        c0_percentile : int or float, default 0
            percentile of participant-measured co2 to use as background baseline if not using constant background
        min_window_threshold : int, default 30
            minimum length of constinuous measurements
        min_co2_threshold : int or float, default 600
            minimum nightly average co2 concentration (ppm) that must be measured
        delta_co2_threshold : int or float, default None
            minimum change in co2 during periods to consider
        truncate_threshold : int, default 0
            minimum difference between subsequent CO2 measurements (ppm) that must be met before considering the remaining measurements.
            Default value of 0 means all data from the increasing period will be used i.e. no truncation.
        decreasing_increase : boolean, default False
            whether to use only the decreasing increase subperiods
        decreasing_increase_rate : int, default 2
            downsample rate used in the decreasing increase method
        plot : boolean, default False
            plot diagnostic plots of each identified period
        save_plot : boolean, default False
            save diagnostic plots 

        Returns
        -------
        buildup_df : DataFrame
            ventilation estimates for each increasing period for each participant from the dynamic model
        """
        buildup_df = pd.DataFrame()
        for pt in beacon_data['beiwe'].unique():
            res_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'ending_co2_meas':[],'ending_co2_calculated':[],'rmse':[],'ach':[]}
            # getting pt-specific data
            beacon_co2_pt = beacon_data[beacon_data['beiwe'] == pt]
            if constant_c0:
                C0 = 450
            else: # pt-based
                C0 = np.nanpercentile(self.beacon_all[self.beacon_all["beiwe"] == pt]["co2"],c0_percentile) 

            info_pt = info[info.index == pt]
            # getting the increasing periods
            increasing_co2_ac_pt = self.get_co2_periods(beacon_co2_pt,window=min_window_threshold,change='increase')
            for period in increasing_co2_ac_pt['period'].unique():
                increasing_period_ac_pt = increasing_co2_ac_pt[increasing_co2_ac_pt['period'] == period]
                # post-processing periods

                ## getting overall change in measured co2
                min_co2_period = np.nanmin(increasing_period_ac_pt['co2'])
                max_co2_period = np.nanmax(increasing_period_ac_pt['co2'])
                delta_co2_period = max_co2_period - min_co2_period

                ## truncating beginning series of measurements
                starting_index = 0
                while increasing_period_ac_pt["change"].iloc[starting_index] <= truncate_threshold and starting_index < len(increasing_period_ac_pt)-1:
                    starting_index += 1

                increasing_period_ac_pt = increasing_period_ac_pt.iloc[starting_index:,:]
                ## getting only periods with decreasing increases in concentration 
                if decreasing_increase:
                    increasing_period_ac_pt = self.get_decreasing_increase_subperiod(increasing_period_ac_pt,decreasing_increase_rate)

                if min_co2_period >= min_co2_threshold and delta_co2_period >= delta_co2_threshold:
                    if "v" in kwargs.keys():
                        V = kwargs["v"]
                    else:
                        V = info_pt['volume'].values[0]
                    if "e" in kwargs.keys():
                        E = kwargs["e"]
                    else:
                        T = np.nanmean(increasing_period_ac_pt['temperature_c'])
                        E = self.get_emission_rate(info_pt.loc[pt,'bmr'],T+273)

                    ach, ss, C_est = self.get_ach_from_dynamic_co2(increasing_period_ac_pt,E,V,C0=C0,pt=pt,period=period,plot=plot,method="build-up",save=save_plot)
                    # adding information to dict
                    for key, value_to_add in zip(res_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                            increasing_period_ac_pt.index[0],increasing_period_ac_pt.index[-1],
                                                            increasing_period_ac_pt['c'][-1],C_est[-1],
                                                            ss,ach]):
                        res_dict[key].append(value_to_add)
                    
            buildup_df = buildup_df.append(pd.DataFrame(res_dict))
            
        return buildup_df

    def compare_decays(self, period_df, ells=[0.1,0.5,1], p=1.0, c0=400.0, measurement_resolution=120, **kwargs):
        """
        Compares measured values to decay estimates
        
        Parameters
        ----------
        period_df : DataFrame
            co2 data from one period
        ells : list, default [0.1,0.5,1]
            test air exchange rates
        p : float, default 1.0
            penetration factor
        c0 : float, default 400.0
            background co2 concentration in ppm
        measurement_resolution : int, default 120
            interval between measurements in seconds
            
        Returns
        -------
        <void>
        """
        # plotting original
        _, ax = plt.subplots(figsize=(12,4))
        ax.plot(period_df.index,period_df["co2"],color="black",lw=3,label="Measured")
        # estimating concentrations and plotting
        ct0 = period_df["co2"][0]
        for ell in ells:
            c = []
            for i in range(len(period_df)):
                t = i*measurement_resolution/3600
                c.append(ct0*math.exp(-ell*t) + (p*c0)*(1 - math.exp(-ell*t)))
            
            ax.plot(period_df.index,c,label=f"ACH = {ell} "+"h$^{-1}$")
            
        if "optimal" in kwargs.keys():
            c_optimal = []
            ell = kwargs["optimal"]
            for i in range(len(period_df)):
                t = i*measurement_resolution/3600
                c_optimal.append(ct0*math.exp(-ell*t) + (p*c0)*(1 - math.exp(-ell*t)))
            
            ax.plot(period_df.index,c_optimal,color="seagreen",lw=2,linestyle="dotted",label=f"ACH = {ell} "+"h$^{-1}$ (optimal)")
                
        ax.legend(frameon=False)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        plt.show()
        plt.close()

class sensitivity_analysis(calculate):
    """
    Sensitivity analysis class for ventilation calculations
    """
    def __init__(self, study="utx000", study_suffix="ux_s20", measurement_resolution=120, morning_hours=3, beacons_to_remove=[17, 40], data_dir="../../data"):
        super().__init__(study, study_suffix, measurement_resolution, morning_hours, beacons_to_remove, data_dir)
        # importing estimates
        self.estimates_ss = pd.read_csv(f"{self.data_dir}/processed/beacon-ventilation_estimates-ss-{self.suffix}.csv",
                        parse_dates=["start","end"],infer_datetime_format=True)
        self.estimates_decay = pd.read_csv(f"{self.data_dir}/processed/beacon-ventilation_estimates-decay-{self.suffix}.csv",
                        parse_dates=["start","end"],infer_datetime_format=True)
        self.estimates_buildup = pd.read_csv(f"{self.data_dir}/processed/beacon-ventilation_estimates-buildup-{self.suffix}.csv",
                        parse_dates=["start","end"],infer_datetime_format=True)

    def get_data_and_fxn(self,method="ss"):
        """
        Gets relevant data/functions depending on the estimation method provided

        Parameters
        ----------
        method : str, default "ss"
            which estimation method to consider

        Returns
        -------
        estimates : DataFrame

        calc_obj : instance of ventilation.calculate

        f : calc_obj function

        
        """
        if method == "decay":
            estimates = self.estimates_decay.copy()
            calc_obj = dynamic(data_dir="../data")
            temp_df = calc_obj.get_morning_beacon_data(self.beacon_nightly,self.beacon_all,num_hours=3)
            self.beacon_morning = temp_df
            f = calc_obj.ventilation_decay
        elif method == "buildup":
            estimates = self.estimates_buildup.copy()
            calc_obj = dynamic(data_dir="../data")
            f = calc_obj.ventilation_buildup
        else: # ss as default
            estimates = self.estimates_ss.copy()
            calc_obj = steady_state(data_dir="../data")
            f = calc_obj.ventilation_ss

        return estimates, calc_obj, f

    def run_one_way(self,method="ss",params=["v","e","c0"],steps=[0,0.25,0.5,0.75,1],v_limits_apt=[712,1449],v_limits_home=[790,1586],
                        e_limits_f=[0.0023744089795121946, 0.003051852397205989],e_limits_m=[0.0031642829343902427, 0.004059001845141127],
                        c0_limits=[0,4],verbose=False):
        """
        Runs sensitivity analysis on ventilation estimates. See discussion in Notebook 4.1.4 regarding default values

        Parameters
        ----------
        ss : str, default "ss"
            which method to use:
                "decay": decay
                "buildup": buildup
                "ss" or catch-all: steady-state
        params : list of str, default ["v","e","c0"]
            variables to use in the sensitivity analysis - limited to ["v","e","c0"]
        steps : list of float, default [0,0.25,0.5,0.75,1]
            steps within the limits to consider
        v_limits_apt : list of int or float, default [490,1670]
            min and max apartment volumes 
        v_limits_home : list of int or float, default [650,1726]
            min and max home volumes 
        e_limits_f : list of int or float, default [0.0023744089795121946, 0.0035768565853658532]
            min and max co2 emission rates from exhaled breath for females
        e_limits_m : list of int or float, default [0.0031642829343902427 - 0.004156063475609755]
            min and max co2 emission rates from exhaled breath for males
        c0_limits_constant : list of int or float, default [400,500]
            min and max constant background co2 concentrations
        c0_limits_pt : list of int or float, default [0,2]
            min and max percentiles to consider for participant-based background co2
        constant_c0 : boolean, default True
            whether to use constant or pt-based background CO2 limits

        Returns
        -------
        sa_res : DataFrame
            ventilation rates calculated through the sensitivity analysis
        """
        sa_res = pd.DataFrame()
        # determining method
        _, _, f = self.get_data_and_fxn(method=method)
        for pt in self.beacon_nightly["beiwe"].unique():
            # pt specific values
            beacon_pt = self.beacon_nightly[self.beacon_nightly["beiwe"] == pt]
            beacon_pt_morning = self.beacon_morning[self.beacon_morning["beiwe"] == pt]
            info_pt = self.info[self.info.index == pt]
            if verbose:
                print("Participant",pt)
            # determine the limits for each variable
            # volume
            if info_pt["volume"].values[0] == 1188:
                # home
                v_limits = v_limits_home
                if verbose:
                    print("\tHome")
            else:
                # apartment
                v_limits = v_limits_apt
                if verbose:
                    print("\tApt")
            # emission rate
            if info_pt["sex"].values[0].lower().startswith("f"):
                # female
                e_limits = e_limits_f
                if verbose:
                    print("\tFemale")
            else:
                # male
                e_limits = e_limits_m
                if verbose:
                    print("\tMale")
                
            # getting ach for each parameter change
            if method == "decay":
                data_to_use = beacon_pt_morning
            else:
                data_to_use = beacon_pt
            for param in params:
                if verbose:
                    print("\tParam:",param)
                for step in steps:
                    if param == "v":
                        value = v_limits[0] + (v_limits[1] - v_limits[0])*step
                        temp = f(data_to_use,self.info,v=value,constant_c0=False)
                    elif param == "e":
                        value = e_limits[0] + (e_limits[1] - e_limits[0])*step
                        temp = f(data_to_use,self.info,e=value,constant_c0=False)
                    elif param == "c0":
                        p = c0_limits[0] + (c0_limits[1] - c0_limits[0])*step
                        temp = f(data_to_use,self.info,c0_percentile=p,constant_c0=False)
                    else:
                        temp = f(data_to_use,self.info)
                        param = "none"
                        
                    temp["parameter"] = param
                    temp["step"] = step
                    if verbose:
                        #print(temp.head())
                        avg_ach = np.nanmean(temp["ach"])
                        print(f"\t\tStep: {step} - {round(avg_ach,3)}")
                    # saving to aggregate df
                    sa_res = sa_res.append(temp)

        return sa_res

    def compare_sa_to_base(self,sa_results,method="ss",params=["v","e","c0"],steps=[0,0.25,0.75,1],
        plot=False,save=False,annot="",save_dir="../reports/figures/"):
        """
        Aggregates the sensitivity analysis results and compares them to the baseline estimates
        
        Parameters
        ----------
        sa_results : DataFrame
            ventilation estimates for each night from each participant with the parameters adjusted
        base_estimates : DataFrame
            baseline ventilation estimates for each night from each participant
        params : list of str, default ["v","e","c0"]
            parameters to include in the analysis
        steps : list of float, default [0,0.25,0.75,1]
            steps within the limits to consider
        plot : boolean, default False
            whether to plot the summary
        save_plot : boolean, default False
            whether to save the summary plot
        
        Returns
        -------
        <results> : DataFrame
            aggregate percent increase results for each step from each parameter
        """
        if method == "ss":
            base_estimates = self.estimates_ss
        elif method == "decay":
            base_estimates = self.estimates_decay
        elif method == "buildup":
            base_estimates = self.estimates_buildup
        else:
            return
        # setting up results dict
        aggregate_res = {step: [] for step in steps}
        aggregate_res["parameter"] = []
        # getting aggregate results
        for param in params:
            sa_param = sa_results[sa_results["parameter"] == param]
            aggregate_res["parameter"].append(param)
            for step in steps:
                perc_inc = []
                for pt in sa_results["beiwe"].unique():
                    sa_pt = sa_param[sa_param["beiwe"] == pt]
                    estimate_pt = base_estimates[base_estimates["beiwe"] == pt]
                    perc_inc.append((np.nanmean(sa_pt[sa_pt["step"] == step]["ach"]) - np.nanmean(estimate_pt["ach"])) / np.nanmean(estimate_pt["ach"]))
                    
                aggregate_res[step].append(round(np.nanmean(perc_inc),4)*100)

        results = pd.DataFrame(aggregate_res).set_index("parameter")
        if plot:
            _, ax = plt.subplots(figsize=(10,6))
            ax.barh(results.index,results.iloc[:,0],left=results.iloc[:,1],color="firebrick",edgecolor="k",label="-1 Step")
            ax.barh(results.index,results.iloc[:,1],color="goldenrod",edgecolor="k",label="$\\frac{-1}{2}$ Step")
            ax.barh(results.index,results.iloc[:,2],color="cornflowerblue",edgecolor="k",label="$\\frac{+1}{2}$ Step")
            ax.barh(results.index,results.iloc[:,3],color="seagreen",left=results.iloc[:,2],edgecolor="k",label="+1 Step")
            # x-axis
            ax.set_xlabel("Percent Change",fontsize=22)
            # y-axis
            ax.set_yticklabels([self.get_param_name(p) for p in params])
            
            # remainder
            ax.tick_params(labelsize=18)
            ax.legend(frameon=False,fontsize=18)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)

            if save:
                plt.savefig("")
            plt.show()
            plt.close()
        
        return results

    def compare_sa_to_base_pt(self,sa_results,base_estimates,params=["v","e","c0"],beacons_to_exclude=[],
        save=False,annot="",save_dir="../reports/figures/"):
        """
        Aggregates sensitivty analysis outcomes on a participant basis

        Parameters
        ----------

        """
        sa_results = sa_results[~sa_results["beacon"].isin(beacons_to_exclude)]
        base_estimates = base_estimates[~base_estimates["beacon"].isin(beacons_to_exclude)]
        
        _, ax = plt.subplots(figsize=(len(sa_results["beacon"].unique()),5),sharex=True)
        for param, offset, shape in zip(params,[-0.2,0.2,0],[".","+","x"]):
            sa_results_param = sa_results[sa_results["parameter"] == param]
            colors = ["firebrick","goldenrod","cornflowerblue","seagreen"]
            for step, color in zip(sa_results_param["step"].unique(),colors):
                sa_results_step = sa_results_param[sa_results_param["step"] == step]
                sa_results_step_summary = sa_results_step.groupby("beacon").mean()
                device_no = []
                for i in range(len(sa_results_step_summary.index)):
                    device_no.append(i + offset)

                sa_results_step_summary["device"] = device_no
                ax.scatter(sa_results_step_summary["device"],sa_results_step_summary["ach"],marker=shape,color=color,edgecolor=color,zorder=10,label=step)

            device_no = []
            for i in base_estimates["beacon"]:
                    device_no.append(i)

            base_estimates["device"] = device_no
            sns.boxplot(base_estimates["device"],base_estimates["ach"],color="white",whis=100,width=0.9,showcaps=False,zorder=1,ax=ax)

            # xlabel
            ax.set_xlabel("ID of Participants with Sufficient Data",fontsize=22)
            ax.set_xticks(sa_results_step_summary["device"])
            label = []
            for bb in sa_results_step_summary.index:
                #label.append("")
                if int(bb) < 10:
                    label.append("0"+str(int(bb)))
                else:
                    label.append(str(int(bb)))
                #label.append("")
            ax.set_xticklabels(label)
            # ylabel
            ax.set_ylabel("Ventilation Rate (h$^{-1}$)",fontsize=22)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=18,)
            # other
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            #ax.legend(frameon=False,fontsize=16,title_fontsize=18,facecolor="white",loc="upper center",bbox_to_anchor=(0.2,1))
            #ax.get_legend().remove()

        if save:
            plt.savefig(f"{save_dir}beacon_summary/ach-sensitivity_by_pt{annot}.pdf")
        plt.show()
        plt.close()

    def traverse_param_range(self,method="ss",steps=[0,0.25,0.5,0.75,1],v_limits_apt=[712,1448],v_limits_home=[790,1586],
        e_limits_f=[0.0023744089795121946, 0.003051852397205989],e_limits_m=[0.0031642829343902427, 0.004059001845141127],
        c0_limits=[0,4],verbose=False):
        """
        Computes ventilation rates for all combinations of parameter steps

        Parameters
        ----------

        Returns
        -------

        """
        sa_res = pd.DataFrame()
        # determining method
        _, _, f = self.get_data_and_fxn(method=method)
        for bb in self.beacon_nightly["beacon"].unique():
            # pt specific values
            beacon_pt = self.beacon_nightly[self.beacon_nightly["beacon"] == bb]
            try:
                beacon_pt_morning = self.beacon_morning[self.beacon_morning["beacon"] == bb]
            except AttributeError:
                if verbose:
                    print("No morning Beacon data available")
            info_pt = self.info[self.info["beacon"] == bb]
            if verbose:
                print("Beacon",bb)
            # determine the limits for each variable
            # volume
            if info_pt["volume"].values[0] == 1188:
                # home
                v_limits = v_limits_home
                if verbose:
                    print("\tHome")
            else:
                # apartment
                v_limits = v_limits_apt
                if verbose:
                    print("\tApt")
            # emission rate
            if info_pt["sex"].values[0].lower().startswith("f"):
                # female
                e_limits = e_limits_f
                if verbose:
                    print("\tFemale")
            else:
                # male
                e_limits = e_limits_m
                if verbose:
                    print("\tMale")
                
            # getting ach for each parameter change
            if method == "decay":
                data_to_use = beacon_pt_morning
            else:
                data_to_use = beacon_pt
            for step1 in steps:
                p = c0_limits[0] + (c0_limits[1] - c0_limits[0])*step1
                for step2 in steps:
                    v_value = v_limits[0] + (v_limits[1] - v_limits[0])*step2
                    for step3 in steps:
                        e_value = e_limits[0] + (e_limits[1] - e_limits[0])*step3
                        if verbose:
                            print(f"\t\tC0: {step1} - {p}\n\t\tV: {step2} - {v_value}\n\t\tE: {step3} - {e_value}")
                        if method in ["buildup","decay"]:
                            temp = f(data_to_use,self.info,
                                constant_c0=False,c0_percentile=p,min_window_threshold=30, min_co2_threshold=600, delta_co2_threshold=120,
                                v=v_value)
                        else: #ss
                            temp = f(data_to_use,self.info,c0_percentile=p,v=v_value,e=e_value,constant_c0=False)
                    
                        for param, step in zip(["c0_step","v_step","e_step"],[step1,step2,step3]): #hard-coded :(
                            temp[param] = step
                        
                        if verbose:
                            #print(temp.head())
                            avg_ach = np.nanmean(temp["ach"])
                            print(f"\t\tStep: {step} - {round(avg_ach,3)}")
                        # saving to aggregate df
                        sa_res = sa_res.append(temp)

        return sa_res

    def traverse_param_percents(self,method="ss",percents=[-0.2,-0.1,0,0.1,0.2],v_limits_apt=[712,1448],v_limits_home=[790,1586],
        e_limits_f=[0.0023744089795121946, 0.0035768565853658532],e_limits_m=[0.0031642829343902427, 0.004156063475609755],
        c0_percentiles=[0,2,4],verbose=False):
        """
        Computes ventilation rates for all combinations of parameter steps

        Parameters
        ----------

        Returns
        -------

        """
        sa_res = pd.DataFrame()
        # determining method
        _, _, f = self.get_data_and_fxn(method=method)
        for bb in self.beacon_nightly["beacon"].unique()[:3]:
            # pt specific values
            beacon_pt = self.beacon_nightly[self.beacon_nightly["beacon"] == bb]
            try:
                beacon_pt_morning = self.beacon_morning[self.beacon_morning["beacon"] == bb]
            except AttributeError:
                if verbose:
                    print("No morning Beacon data available")
            info_pt = self.info[self.info["beacon"] == bb]
            if verbose:
                print("Beacon",bb)
            # determine the limits for each variable
            # volume
            if info_pt["volume"].values[0] == 1188:
                # home
                v_limits = v_limits_home
                if verbose:
                    print("\tHome")
            else:
                # apartment
                v_limits = v_limits_apt
                if verbose:
                    print("\tApt")
            # emission rate
            if info_pt["sex"].values[0].lower().startswith("f"):
                # female
                e_limits = e_limits_f
                if verbose:
                    print("\tFemale")
            else:
                # male
                e_limits = e_limits_m
                if verbose:
                    print("\tMale")
                
            # getting ach for each parameter change
            if method == "decay":
                data_to_use = beacon_pt_morning
            else:
                data_to_use = beacon_pt
            for c0_percentile in c0_percentiles:
                for p1 in percents:
                    v_value = np.mean(v_limits) * (1+p1)
                    for p2 in percents:
                        e_value = np.mean(e_limits) * (1+p2)
                        if verbose:
                            print(f"\t\tC0: {c0_percentile}\n\t\tV: {p1*100}%\n\t\tE: {p2*100}%")
                        if method in ["buildup","decay"]:
                            temp = f(data_to_use, self.info,
                                constant_c0=False, c0_percentile=c0_percentile, min_window_threshold=30, min_co2_threshold=600, delta_co2_threshold=120,
                                e=e_value, v=v_value)
                        else: #ss
                            temp = f(data_to_use,self.info,c0_percentile=c0_percentile,v=v_value,e=e_value,constant_c0=False)
                    
                        for param, step in zip(["c0_step","v_step","e_step"],[c0_percentile,p1,p2]): #hard-coded :(
                            temp[param] = step
                        
                        if verbose:
                            avg_ach = np.nanmean(temp["ach"])
                            print(f"\tACH: {round(avg_ach,3)}")
                        # saving to aggregate df
                        sa_res = sa_res.append(temp)

        return sa_res

    def param_range_heatmap(self,data,method="ss",params=("c0","v","e"),plot=False,save=False,save_dir="../reports/figures",**kwargs):
        """
        Creates a heatmap of the percent increases for parameter changes

        Parameters
        ----------
        data : DataFrame
            comprehensive dataframe containing all participant ventilation rates for each parameter change
        param_order : tuple of str, default ["c0","v","e"]
            order of parameters to consider when creating heatmaps
        plot : boolean, default False
            whether to plot the heatamps or not
        save : boolean, default False
            whether to save the heatmaps or not

        Returns
        -------
        percent_increases : dict of DataFrame
            dataframes corresponding to each heatmap with keys corresponding to the first parameter steps
        """
        if method.lower() == "ss":
            estimates = self.estimates_ss.copy()
        elif method.lower() == "buildup":
            estimates = self.estimates_buildup.copy()
        elif method.lower() == "decay":
            estimates = self.estimates_decay.copy()
        else:
            print("invalid method - exiting")
            return

        percent_increases = {}
        # combining sa results with base rates to get percent increase
        data_comb = data.merge(estimates[["beacon","start","end","ach"]],
                            on=["beacon","start","end"],how="left",suffixes=["_sa","_base"])
        data_comb["percent_increase"] = (data_comb["ach_sa"] - data_comb["ach_base"]) / data_comb["ach_base"]
        for i, step in enumerate(data_comb[f"{params[0]}_step"].unique()):
            data_comb_step = data_comb[data_comb[f"{params[0]}_step"] == step] # subselecting
            data_summary = data_comb_step.groupby([f"{params[1]}_step",f"{params[2]}_step"]).mean() # getting aggregate percent increases
            df_to_plot = data_summary.reset_index().pivot(index=f"{params[1]}_step",columns=f"{params[2]}_step",values="percent_increase") # formatting for heatmap
            df_to_plot.sort_index(ascending=False,inplace=True)
            percent_increases[f"{params[0]}_step_{i}"] = df_to_plot # saving

            if plot:
                _, ax = plt.subplots(figsize=(6,6))
                if "vmin" in kwargs.keys():
                    vmin = kwargs["vmin"]
                else:
                    vmin = -1
                if "vmax" in kwargs.keys():
                    vmax = kwargs["vmax"]
                else:
                    vmax = 1
                    
                for c in df_to_plot.columns:
                    df_to_plot[c] = np.where(abs(df_to_plot[c]) < 0.005, 0, df_to_plot[c])

                hm = sns.heatmap(df_to_plot,
                        vmin=vmin, vmax=vmax, center=0, 
                        cmap=sns.diverging_palette(20, 220, n=200),cbar_kws={'ticks':np.arange(vmin,vmax+0.5,0.5),"shrink":0.9},fmt=".0%",
                        square=True,linewidths=1,annot=True,annot_kws={"size":12},ax=ax)
                # x-axis
                ax.set_xlabel(self.get_param_name(params[2]),fontsize=16)
                ax.set_xticklabels([f"{int(float(t.get_text())*100)}%" for t in ax.get_xticklabels()])
                # y-axis
                ax.set_ylabel(self.get_param_name(params[1]),fontsize=16)
                ax.set_yticklabels([f"{int(float(t.get_text())*100)}%" for t in ax.get_yticklabels()])
                # remainder
                cbar = hm.collections[0].colorbar
                cbar.ax.tick_params(labelsize=12)
                ax.tick_params(labelsize=13)
                ax.set_title(f"{self.get_param_name(params[0])}: {step}th Percentile",fontsize=16)

                if save:
                    plt.savefig(f"{save_dir}/ventilation-param_combination-{method}-heatmap-{params[0]}_step_{i}.pdf")
                plt.show()
                plt.close()

        return percent_increases

    def run_sobol(self,method="ss",params=["v","e","c0"],v_limits=[712,1586],e_limits=[0.0023744089795121946, 0.004059001845141127],c0_limits=[0,5]):
        """
        Gets Sobol Indices

        Parameters
        ----------
        method : str, default "ss"

        Returns
        -------
        """
        # determining method
        if method in ["buildup","decay",]:
            calc_obj = dynamic(data_dir="../data")
            f = calc_obj.get
        else: # ss as default
            calc_obj = steady_state(data_dir="../data")
            f = calc_obj.get_ach_from_constant_co2
        
        #res = pd.DataFrame()

        problem = {
            'num_vars': len(params),
            'names': params,
            'bounds': [v_limits,
                    e_limits,
                    c0_limits]
        }

        param_values = saltelli.sample(problem, 10000)
        
        Y = np.zeros([param_values.shape[0]])
        for i, X in enumerate(param_values):
            E = X[1]
            V = X[0]
            C0 = X[2]
            C = 1000 # some dummy value
            Y[i] = f(E,V,C,C0)

        Si = sobol.analyze(problem, Y, calc_second_order=False)

        total_Si, first_Si = Si.to_df()
        return total_Si, first_Si

def plot_strip(summarized_rates, sort_order=[True, False], conduct_ttest=False, 
    save=False, annot="", save_dir="../reports/figures/"):
    """
    Strip plots of ventilation rates

    Parameters
    ----------
    summarized_rates : DataFrame
        ventilation rates for each method and participant
    sort_order : list of boolean, default [True, False]
        how to sort the device number and then method name
    conduct_ttest : boolean, default False
        whether to conduct a ttest on the data or not
    save : boolean, default False
        whether to save the figure or not
    annot : str, default ""
        extran annotations to include in filename
    save_dir : str, default "../reports/figures/"
        path to save location

    Returns
    -------
    ttest_res : DataFrame
        results from conducting the ttest
    """
    _, ax = plt.subplots(figsize=(14,6))
    df_to_plot = summarized_rates.copy()
    device_no = []
    for bb in df_to_plot["beacon"]:
        if int(bb) < 10:
            device_no.append("0"+str(int(bb)))
        else:
            device_no.append(str(int(bb)))

    df_to_plot["device"] = device_no
    df_to_plot.sort_values(["device","method"],ascending=sort_order,inplace=True)
    df_to_plot["method_title"] = ["Steady-State" if method.startswith("ss") else method.split("_")[0].title() for method in df_to_plot["method"]]
    sns.stripplot(x="device",y="ach",hue="method_title",
        palette=["white",'#bf5700','navy',"goldenrod","firebrick"],edgecolor="black",linewidth=1,size=8,jitter=0.25,alpha=0.5,data=df_to_plot,ax=ax)
    # xlabel
    ax.set_xlabel("ID of Participants with Sufficient Data",fontsize=22)
    plt.xticks(fontsize=18)
    # ylabel
    ax.set_ylabel("Ventilation Rate (h$^{-1}$)",fontsize=22)
    ax.set_ylim(bottom=0)
    plt.yticks(fontsize=18)
    # other
    for loc in ["top","right"]:
        ax.spines[loc].set_visible(False)
    ax.legend(frameon=False,fontsize=16,title_fontsize=18,facecolor="white",loc="upper center",bbox_to_anchor=(0.2,1))
    #ax.get_legend().remove()

    if conduct_ttest and len(df_to_plot["method"].unique()) == 2:
        # creating results dict
        ttest_res = {method: [] for method in df_to_plot["method"].unique()}
        ttest_res["device"] = []
        ttest_res["p"] = []
        # running t-test is possible
        for device in df_to_plot["device"].unique():
            data_device = df_to_plot[df_to_plot["device"] == device]
            methods = data_device["method"].unique()
            m0 = data_device[data_device["method"] == methods[0]]
            m1 = data_device[data_device["method"] == methods[1]]
            if len(m0) >= 10 and len(m1) >= 10: # only considering participants with at least 10 or more estiamtes for each method
                _, p = ttest_ind(m0["ach"], m1["ach"], equal_var=False, nan_policy="omit")
                for key, val in zip(ttest_res.keys(),[len(m0),len(m1),device,p]):
                    ttest_res[key].append(val)

        # annotating
        for device, p in zip(ttest_res["device"],ttest_res["p"]):
            weight="bold" if p < 0.05 else "normal"
            print(device)
            print(p)
            #ax.text(str(device),ax.get_ylim()[1],round(p,3),weight=weight)


    if save:
        plt.savefig(f"{save_dir}/beacon_summary/ventilation_rates{annot}-strip-ux_s20.pdf",bbox_inches="tight")
        
    plt.show()
    plt.close()

    if conduct_ttest:
        return ttest_res

def plot_distribution(summarized_rates,save=False, save_dir="../reports/figures/"):
    """
    Plots the distribution of ventilation rates

    Parameters
    ----------
    summarized_rates : DataFrame
        ventilation rates for each method and participant
    save : boolean, default False
        whether to save the figure or not
    save_dir : str, default "../reports/figures/"
        path to save location

    Returns
    -------
    <void>
    """
    df_to_plot = summarized_rates.copy()
    device_no = []
    for bb in df_to_plot["beacon"]:
        if bb < 10:
            device_no.append("0"+str(int(bb)))
        else:
            device_no.append(str(int(bb)))

    df_to_plot["device"] = device_no
    df_to_plot.sort_values("device",inplace=True)
    df_to_plot["method_title"] = ["Steady-State" if method.startswith("ss") else method.replace("_"," ").title() for method in df_to_plot["method"]]
    colors = ['black','#bf5700','navy',"gray","goldenrod","firebrick"][:len(df_to_plot["method"].unique())]
    _, ax = plt.subplots(figsize=(12,4))
    sns.kdeplot(x="ach",hue="method_title",palette=colors,cut=0,data=df_to_plot,ax=ax)
    # xlabel
    ax.set_xlabel("Ventilation Rate (h$^{-1}$)",fontsize=18)
    plt.xticks(fontsize=14)
    # ylabel
    ax.set_ylabel("Density",fontsize=18)
    plt.yticks(fontsize=14)
    # other
    for loc in ["top","right"]:
        ax.spines[loc].set_visible(False)
    #ax.legend(frameon=False,title="Estimation Method",fontsize=13,title_fontsize=16)

    if save:
        plt.savefig(f"{save_dir}/beacon_summary/ventilation_rates-dist-ux_s20.pdf",bbox_inches="tight")
    plt.show()
    plt.close()

def tabulate_estimates(summarized_rates):
    """
    Creates a table of summarized ventilation estimates
    """
    res = {"method":[],"number":[],"number_of_participants":[],"min":[],"25%":[],"median":[],"75%":[],"90%":[],"max":[],"mean":[]}

    # per method
    for method in summarized_rates["method"].unique():
        # method specific data
        rates_method = summarized_rates[summarized_rates["method"] == method]
        method_title = "Steady-State" if method.startswith("ss") else method.replace("_"," ").title()
        n = len(rates_method)
        n_pt = len(rates_method["beacon"].unique())
        for k, v in zip(res.keys(),[method_title,n,n_pt,np.nanmin(rates_method["ach"]),np.nanpercentile(rates_method["ach"],25),np.nanmedian(rates_method["ach"]),np.nanpercentile(rates_method["ach"],75),np.nanpercentile(rates_method["ach"],90),np.nanmax(rates_method["ach"]),
                np.nanmean(rates_method["ach"])]):
            try:
                res[k].append(round(v,2))
            except TypeError:
                # can't round a str
                res[k].append(v)

    # aggregate
    for k, v in zip(res.keys(),["Aggregate",len(summarized_rates),len(summarized_rates["beacon"].unique()),
            np.nanmin(summarized_rates["ach"]),np.nanpercentile(summarized_rates["ach"],25),np.nanmedian(summarized_rates["ach"]),np.nanpercentile(summarized_rates["ach"],75),np.nanpercentile(summarized_rates["ach"],90),np.nanmax(rates_method["ach"]),
            np.nanmean(summarized_rates["ach"])]):
        try:
            res[k].append(round(v,2))
        except TypeError:
            # can't round a str
            res[k].append(v)

    df = pd.DataFrame(res)
    df.columns = [col.replace("_"," ").title() for col in df.columns]
    df.set_index("Method",inplace=True)
    print(df.to_latex())
    return df

def perform_t_test(a,b,equal_var=True):
    """
    Performs a t-test on the means between two groups
    """
    t, p = ttest_ind(a,b,equal_var=equal_var)
    return t, p

def perform_kruskal_test(a,b,equal_var=True):
    """
    Performs a test on the medians between two groups
    """
    h, p = kruskal(a,b,nan_policy="omit")
    return h, p

def transform_data(data,power=1):
    """
    Box-Cox transformation on the data
    """
    return boxcox(data,power)

def main():
    ventilation_estimate = calculate()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='ventilation.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()