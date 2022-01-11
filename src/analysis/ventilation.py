# Ventilation 
# -----------
# Author: Hagen
# Date: 03/29/21
# Description: 
# Data Files Needed:
#   - beacon
#   - beacon-fb_and_gps_filtered
#   - 

import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from datetime import datetime, timedelta
import math

class calculate():

    def __init__(self, study="utx000", study_suffix="ux_s20", measurement_resolution=120, morning_hours=3, data_dir="../../data"):
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

        # fitbit data
        self.daily_act = pd.read_csv(f'{self.data_dir}/processed/fitbit-daily-{self.suffix}.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
        weight_dict = {'beiwe':[],'mass':[]}
        for pt in self.daily_act['beiwe'].unique():
            daily_pt = self.daily_act[self.daily_act['beiwe'] == pt]
            weight_dict['beiwe'].append(pt)
            weight_dict['mass'].append(np.nanmean(daily_pt['weight'])*0.453592) # convert to kg
        self.sleep_summary = pd.read_csv(f"{self.data_dir}/processed/fitbit-sleep_summary-{self.suffix}.csv",parse_dates=["start_time","end_time","start_date","end_date"],infer_datetime_format=True)
            
        # participant information
        pt_names = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='all')
        pt_names = pt_names[["beiwe","first","last","sex"]]
        pt_ids = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='beacon')
        pt_ids = pt_ids[['redcap','beiwe','beacon','lat','long','volume','roommates']] # keep their address locations
        info = pt_ids.merge(right=pt_names,on='beiwe')
        mass_info = info.merge(left_on='beiwe',right=pd.DataFrame(weight_dict),right_on='beiwe')
        mass_info['bmr'] = mass_info.apply(lambda row: self.get_BMR(row['sex'],row['mass']),axis=1)
        self.info = mass_info.set_index('beiwe')

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

    def get_co2_periods(self, beacon_data, window=30, co2_threshold=10, t_threshold=0.25, time_threshold=120, difference_threshold=30, change='decrease'):
        '''
        Finds and keeps periods of CO2 change or consistency
        
        Parameters
        ----------
        beacon_data : DataFrame
            measured CO2 concentrations
        window : int, default 30
            many timesteps the increase/decrease period has to last
        co2_threshold : int or float, default 10
            tolerance on the variance in co2 concentration in ppm
        t_threshold : float, default 0.25
            tolerance on temperature variation in C
        time_threshold : float of int, default 120
            maximum time difference, in seconds, between subsequent measurements
        difference_threshold : int or float, default 30 (resolution on SCD30 sensor)
            minimum co2 difference, in ppm, that must occur over the period to be considered - only used when change in ["increase","decrease"]
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
                while df['change'][i] > 0 and abs(df['t_change'][i]) <= t_threshold and df['dtime'][i].total_seconds() <= time_threshold+2:
                    periods.append(period)
                    i += 1

                periods.append(0)
                period += 1
                i += 1
        else: # constant periods
            while i < len(df):
                while abs(df['change'][i]) < co2_threshold and df['t_change'][i] <= 0 and df['dtime'][i].total_seconds() <= time_threshold+2:
                    periods.append(period)
                    i += 1

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
            # difference in concentrations too low
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

    def run_sensitivity_analysis(self,ss=True,params=["v","e","c0"],steps=[0,0.25,0.5,0.75,1],v_limits_apt=[490,1670],v_limits_home=[650,1726],
                            e_limits_f=[0.0023744089795121946, 0.0035768565853658532],e_limits_m=[0.0031642829343902427, 0.004156063475609755],
                            c0_limits_constant=[400,500],c0_limits_pt=[0,2],constant_c0=True,verbose=False):
        """
        Runs sensitivity analysis on ventilation estimates. See discussion in Notebook 4.1.4 regarding default values

        Parameters
        ----------
        ss : boolean, default True
            whether to use steady_state estimate or decay (eq 2)
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
            ventilation rates calculated throught the sensitivity analysis
        """
        sa_res = pd.DataFrame()
        for pt in self.beacon_nightly["beiwe"].unique():
            # pt specific values
            beacon_pt = self.beacon_nightly[self.beacon_nightly["beiwe"] == pt]
            beacon_pt_all = self.beacon_all[self.beacon_all["beiwe"] == pt]
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
            # determining steady-state or decay estimation method
            if ss:
                sa = steady_state(data_dir="../data")
                f = sa.ventilation_ss
            else: #decay
                sa = decay(data_dir="../data")
                f = sa.ventilation_decay_no_occupant
            # getting ach for each parameter change
            for param in params:
                if verbose:
                    print("\tParam:",param)
                for step in steps:
                    if param == "v":
                        value = v_limits[0] + (v_limits[1] - v_limits[0])*step
                        temp = f(beacon_pt,self.info,v=value,constant_c0=constant_c0)
                    elif param == "e":
                        value = e_limits[0] + (e_limits[1] - e_limits[0])*step
                        temp = f(beacon_pt,self.info,e=value,constant_c0=constant_c0)
                    elif param == "c0":
                        if constant_c0:
                            value = c0_limits_constant[0] + (c0_limits_constant[1] - c0_limits_constant[0])*step
                            temp = f(beacon_pt,self.info,c0=value,constant_c0=constant_c0)
                        else:
                            p = c0_limits_pt[0] + (c0_limits_pt[1] - c0_limits_pt[0])*step
                            value = np.nanpercentile(beacon_pt_all["co2"],p)
                            temp = f(beacon_pt,self.info,c0=value,constant_c0=constant_c0)
                    else:
                        temp = f(beacon_pt,self.info)
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

    def ventilation_ss(self, beacon_data, info, constant_c0=True, c0_percentile=1, data_length_threshold=90, min_window_threshold=30, min_co2_threshold=600, 
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
            minimum length of constinuous measurements
        min_co2_threshold : int or float, default 600
            minimum nightly average co2 concentration that must be measured
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
        for pt in beacon_data['beiwe'].unique(): # cycling through each of the participants
            # setting up the dictionary to add pt values to
            pt_dict = {'beiwe':[],'beacon':[],'start':[],'end':[],'co2_mean':[],'co2_delta':[],'t_mean':[],'t_delta':[],'e':[],'c0':[],"v":[],'ach':[]}
            # pt-specific dataframes
            beacon_pt = beacon_data[beacon_data['beiwe'] == pt]
            if constant_c0:
                C0 = 450
            else: # pt-based
                C0 = np.nanpercentile(self.beacon_all[self.beacon_all["beiwe"] == pt]["co2"],c0_percentile)
            info_pt = info[info.index == pt]

            # looping through pt-specific sleep events registered by FB
            for start, end in zip(beacon_pt['start_time'].unique(),beacon_pt['end_time'].unique()): 
                beacon_pt_night = beacon_pt[start:end] # masking for iaq data during sleep
                if len(beacon_pt_night) > data_length_threshold:
                    # getting constant periods per sleep event
                    constant_periods = self.get_co2_periods(beacon_pt_night[['co2','temperature_c','rh']],window=min_window_threshold,change='constant')
                    if len(constant_periods) > 0:
                        # summarizing the constant period(s)
                        for period in constant_periods['period'].unique():
                            constant_by_period = constant_periods[constant_periods['period'] == period]
                            C = np.nanmean(constant_by_period['co2'])
                            if C > min_co2_threshold: # must have measured co2 over a certain threshold to be considered
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
                                if "c0" in kwargs.keys():
                                    C0 = kwargs["c0"]
                                else:
                                    pass # value has already been established outside of this loop
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

    def plot_constant_co2_period(df, ach, pt="", save=False):
        """
        plots the co2 and temperature from a constant co2 period
        
        Parameters
        ----------
        df : DataFrame
        
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
        ax.set_title(f"ID: {pt} - Period: {period} - ACH: {round(ach,2)}")
        
        if save:
            plt.savefig(f"../reports/figures/beacon_summary/ventilation_estimates/method_0-{pt}-{period}.pdf",bbox_inches="tight")
            
        plt.show()
        plt.close()

class decay(calculate):

    def get_morning_beacon_data(self, night_df,all_df,num_hours=3):
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

    def get_ach_from_dynamic_co2(self, df, E, V, C0=450.0, p=1.0, measurement_resolution=120, plot=False, pt="", period="", method="", save=False):
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
        plot : boolean, default False
            whether to plot the diagnostic decay periods
        pt : str, default ""
            participant beiwe id
        period : str, default ""
            period number (used for diagnostic plot)
        method : str, default ""

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
        min_rmse = math.inf
        ach = -1
        C_to_plot = df['c'].values # for comparison
        # looping through possible ach values
        for ell in np.arange(0,10.001,0.001):
            Cs = []
            for i in range(len(df)):
                t = i*measurement_resolution/3600
                Cs.append(C_t0 * math.exp(-ell*t) + (p*C0_gm3 - E_gh/(V_m3*ell))*(1 - math.exp(-ell*t)))
                
            # calculating error metric(s)
            # we use rmse here because it tends to penalize larger differences more than mae
            rmse = 0
            for C_est, C_meas in zip(Cs,df['c']):
                rmse += (C_est-C_meas)**2
            rmse = math.sqrt(rmse/len(Cs))

            # saving best result
            if rmse < min_rmse:
                min_rmse = rmse
                ach = ell
                C_to_plot = Cs
                
        # plotting to compare results
        if plot:
            _, ax = plt.subplots(figsize=(8,6))
            # concentration axis
            ax.plot(df.index,df['c'],color='seagreen',label='Measured')
            ax.plot(df.index,C_to_plot,color='firebrick',label=f'ACH={round(ach,2)}; RMSD={round(rmse,3)}')

            for i in range(len(Cs)):
                ax.annotate(str(round(df['c'].values[i],2)),(df.index[i],df['c'].values[i]),ha="left",fontsize=12)
                ax.annotate(str(round(C_to_plot[i],2)),(df.index[i],C_to_plot[i]),ha="right",fontsize=12)
                
            ax.set_ylabel("CO$_2$ (g/m$^3$)",fontsize=16)
            plt.yticks(fontsize=14)
            ax.legend(fontsize=14)
            plt.xticks(fontsize=14,ha="left",rotation=-15)

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
            
        return ach, min_rmse, C_to_plot

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

    def ventilation_decay_no_occupant(self, beacon_data, info, constant_c0=True, c0_percentile=1, min_window_threshold=30, min_co2_threshold=600, plot=False, save_plot=False,**kwargs):
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
        c0_percentile : int or float, default 1
            percentile of participant-measured co2 to use as background baseline if not using constant background
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
            if "c0" in kwargs.keys():
                C0 = kwargs["c0"]
            else:
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
            for period in decreasing_co2_ac_pt['period'].unique():
                decreasing_period_ac_pt = decreasing_co2_ac_pt[decreasing_co2_ac_pt['period'] == period]
                if np.nanmin(decreasing_period_ac_pt['co2']) >= min_co2_threshold:
                    # calculating ventilation
                    ach, ss, C_est = self.get_ach_from_dynamic_co2(decreasing_period_ac_pt,E=0,V=V,C0=C0,plot=plot,pt=pt,period=period,method=2,save=save_plot)
                    # adding information to dict
                    for key, value_to_add in zip(decay_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                            decreasing_period_ac_pt.index[0],decreasing_period_ac_pt.index[-1],
                                                            decreasing_period_ac_pt['c'][-1],C_est[-1],
                                                            ss,ach]):
                        decay_dict[key].append(value_to_add)
                    
            decay_df = decay_df.append(pd.DataFrame(decay_dict))

        return decay_df

    def ventilation_decay_full(self, beacon_data, info, min_window_threshold=30, min_co2_threshold=600, plot=False, save_plot=False):
        """
        Ventilation estimate based on no indoor sources

        Parameters
        ----------
        beacon_data : DataFrame
            contains beacon data from the morning after participants wake
        info : DataFrame
            participant demographic info
        min_window_threshold : int, default 30
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
            C0 = np.nanpercentile(self.beacon_all[self.beacon_all["beiwe"] == pt]["co2"],5) # setting background to the 5th percentile co2 concentration
            info_pt = info[info.index == pt]
            # getting 
            decreasing_co2_ac_pt = self.get_co2_periods(beacon_co2_pt,window=min_window_threshold,change='decrease')
            for period in decreasing_co2_ac_pt['period'].unique():
                decreasing_period_ac_pt = decreasing_co2_ac_pt[decreasing_co2_ac_pt['period'] == period]
                if np.nanmin(decreasing_period_ac_pt['co2']) >= min_co2_threshold:
                    T = np.nanmean(decreasing_period_ac_pt['temperature_c'])
                    E = self.get_emission_rate(info_pt.loc[pt,'bmr'],T+273)
                    V = info_pt['volume'].values[0]
                    ach, ss, C_est = self.get_ach_from_dynamic_co2(decreasing_period_ac_pt,E,V,C0=C0,pt=pt,period=period,plot=plot,method=1,save=save_plot)
                    # adding information to dict
                    for key, value_to_add in zip(decay_dict.keys(),[pt,info_pt['beacon'].values[0],
                                                            decreasing_period_ac_pt.index[0],decreasing_period_ac_pt.index[-1],
                                                            decreasing_period_ac_pt['c'][-1],C_est[-1],
                                                            ss,ach]):
                        decay_dict[key].append(value_to_add)
                    
            decay_df = decay_df.append(pd.DataFrame(decay_dict))
            
        return decay_df

def plot_strip(summarized_rates, save=False, save_dir="../reports/figures/"):
    """
    Plots strip plots of ventilation rates
    """
    
    _, ax = plt.subplots(figsize=(14,6))
    df_to_plot = summarized_rates.copy()
    device_no = []
    for bb in df_to_plot["beacon"]:
        if bb < 10:
            device_no.append("0"+str(int(bb)))
        else:
            device_no.append(str(int(bb)))

    df_to_plot["device"] = device_no
    df_to_plot.sort_values("device",inplace=True)
    df_to_plot["method_title"] = ["Steady-State" if method.startswith("ss") else method.split("_")[0].title() for method in df_to_plot["method"]]
    sns.stripplot(x="device",y="ach",hue="method_title",palette=['black','#bf5700','navy',"gray"],size=10,jitter=0.2,alpha=0.7,data=df_to_plot,ax=ax)
    # xlabel
    ax.set_xlabel("Device Number",fontsize=22)
    plt.xticks(fontsize=18)
    # ylabel
    ax.set_ylabel("Ventilation Rate (h$^{-1}$)",fontsize=22)
    plt.yticks(fontsize=18)
    # other
    for loc in ["top","right"]:
        ax.spines[loc].set_visible(False)
    ax.legend(frameon=True,title="Estimation Method",fontsize=16,title_fontsize=18,facecolor="white")

    if save:
        plt.savefig(f"{save_dir}/beacon_summary/ventilation_rates-strip-ux_s20.pdf",bbox_inches="tight")
        
    plt.show()
    plt.close()

def plot_distribution(summarized_rates,save=False, save_dir="../reports/figures/"):
    _, ax = plt.subplots(figsize=(14,6))
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
    colors = ['black','#bf5700','navy',"gray"][:len(df_to_plot["method"].unique())]
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

def main():
    ventilation_estimate = calculate()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='ventilation.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()