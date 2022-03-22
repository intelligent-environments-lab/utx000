# -*- coding: utf-8 -*-
# Data Science Packages
import pandas as pd
import numpy as np
import scipy.stats as stats

# Useful
from datetime import datetime, timedelta
import math

# Operations
import os

# Extra
import ast
import warnings
warnings.filterwarnings('ignore')

class wcwh():
    """
    Class dedicated to processing data from the wcwh family studies
    """

    def __init__(self,study,suffix,data_dir="../../data",
        beacon_list=np.arange(1,51,1),start_time=datetime(2021,12,1),end_time=datetime(2023,12,1),
        verbose=0):
        """
        Initializing Method

        Parameters
        ----------
        study : str
            specific study name to pull from
        suffix : str
            specific suffix associated with a study
        data_dir : str, default "../../data/"
            path to raw data
        beacon_list : np array, default np.arange(1,51,1)
            list of beacon numbers to use
        verbose : int, default 0
            verbose level where 3 or higher is debugging level
        start_time : Datetime
            beginning data for period to consider data
        end_time : Datetime
            ending data for period to consider data
        """
        # study specifics
        self.study = study
        self.suffix = suffix
        self.data_dir = data_dir

        self.beacon_list = beacon_list
        self.start_time = start_time
        self.end_time = end_time

        self.verbose = verbose

        # beacon correction factors 
        self.correction = {}
        for file in os.listdir(f"{self.data_dir}/interim/{self.study}/"):
            if verbose > 2:
                print(file)
            file_info = file.split("-")
            if len(file_info) == 3:
                if file_info[1] == "linear_model" and file_info[-1] == self.suffix+".csv":
                    try:
                        if verbose > 2:
                            print(f"Reading for {file_info[0]}")
                        self.correction[file_info[0]] = pd.read_csv(f'{self.data_dir}/interim/{self.study}/{file}',index_col=0)
                    except FileNotFoundError:
                        print(f"Missing file for {file_info[0]} - padding with zeros")
                        self.correction[file_info[0]] = pd.DataFrame(data={"beacon":np.arange(1,51),"constant":np.zeros(51),"coefficient":np.ones(51)}).set_index("beacon")

        # EMA Attributes
        self.ema_start = datetime(2020,6,1)
        self.ema_end = datetime(2020,9,1)

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def process_beacon(self,extreme=None,retain_negative=True,retain_na=True,resample_rate=1,
        columns_to_drop=["Visible","visible-unitless","Infrared","infrared-unitless","eCO2","equivalent_carbon_dioxide-ppm"],
        columns_to_leave_raw=[]):
        '''
        Processes data from the beacons

        Parameters
        ----------
        resample_rate : int, default 1
            number of minutes to resample to
        extreme : str, default None
            method to use for outlier detection and removing
            "zscore": remove values outside +/- 2.5 zscore
            "iqr": remove values outside +/- 1.5*IQR
        retain_negative : boolean, default True
            whether to retain negative measurements for IAQ parameters
        retain_na : boolean, default True
            whether to retain NaN values
        columns_to_drop : list of str, default ["Visible","visible-unitless","Infrared","infrared-unitless","eCO2","equivalent_carbon_dioxide-ppm"]
            data that are not needed
        columns_to_leave_raw : list of str, default []
            variables we do not want to correct

        Returns
        -------
        <done> : boolean
            True if able to save dataframe that contains all the data at regular intervals in processed directory
        '''

        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder=f"{self.data_dir}/raw/{self.study}/beacon"
        print("\tProcessing beacon data...")
        if self.verbose > 0:
            print("\n\t\tReading for beacon:")
        for beacon in self.beacon_list:

            # correcting the number since the values <10 have leading zero in directory
            number = f'{beacon:02}'
            if self.verbose > 0:
                print(f'\t\t{number}')

            beacon_folder=f'{beacons_folder}/B{number}/DATA'
            beacon_df = pd.DataFrame() # dataframe specific to the beacon

            df_list = []
            if os.path.isdir(beacon_folder):
                for file in os.listdir(beacon_folder):
                    if file[-1] == "v":
                        y = int(file.split("_")[1].split("-")[0])
                        m = int(file.split("_")[1].split("-")[1])
                        d = int(file.split("_")[1].split("-")[2].split(".")[0])
                        date = datetime(y,m,d)
                        if date.date() >= self.start_time.date() and date.date() <= self.end_time.date():
                            try:
                                # reading in raw data (csv for one day at a time) and appending it to the overall dataframe
                                if self.verbose > 2:
                                    print("\t\t\t",file)
                                try:
                                    day_df = pd.read_csv(f'{beacon_folder}/{file}',
                                                    index_col='Timestamp',parse_dates=True, # old header format
                                                    infer_datetime_format=True)
                                except ValueError:
                                    day_df = pd.read_csv(f'{beacon_folder}/{file}',
                                                    index_col='timestamp',parse_dates=True, # new header format
                                                    infer_datetime_format=True)

                                # dropping unecessary columns
                                for col in columns_to_drop:
                                    try:
                                        day_df.drop([col],axis=1,inplace=True)
                                    except KeyError:
                                        if self.verbose > 1:
                                            print(f"\t\t\tNo column {col} in dataframe")

                                # renaming columns
                                ## old headings
                                day_df.rename(columns={"TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                                        "Temperature [C]":"internal_temp","Relative Humidity":"internal_rh","T_NO2":"t_no2","RH_NO2":"rh_no2","T_CO":"t_co","RH_CO":"rh_co",
                                                        "PM_N_0p5":"pm0p5_number","PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_4":"pm4_number","PM_N_10":"pm10_number",
                                                        "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_4":"pm4_mass","PM_C_10":"pm10_mass"},inplace=True)
                                ## new headings
                                day_df.rename(columns={"total_volatile_organic_compounds-ppb":"tvoc","light-lux":"lux","nitrogen_dioxide-ppb":"no2","carbon_monoxide-ppb":"co","carbon_dioxide-ppm":"co2",
                                                        "t_from_co2-c":"internal_temp","rh_from_co2-percent":"internal_rh","t_from_no2-c":"t_no2","rh_from_no2-percent":"rh_no2","t_from_co-c":"t_co","rh_from_co-percent":"rh_co",
                                                        "pm0p5_count-number_per_cm3":"pm0p5_number","pm1_count-number_per_cm3":"pm1_number","pm2p5_count-number_per_cm3":"pm2p5_number","pm4_count-number_per_cm3":"pm4_number","pm10_count-number_per_cm3":"pm10_number",
                                                        "pm1_mass-microgram_per_m3":"pm1_mass","pm2p5_mass-microgram_per_m3":"pm2p5_mass","pm4_mass-microgram_per_m3":"pm4_mass","pm10_mass-microgram_per_m3":"pm10_mass"},inplace=True)
                                day_df.index.rename("timestamp",inplace=True)
                                
                                df_list.append(day_df)
                                
                            except Exception as e:
                                # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                                print(e)
                                print(f'Issue encountered while importing {beacon_folder}/{file}, skipping...')
                                self.move_to_purgatory(f'{beacon_folder}/{file}',f'{self.data_dir}/purgatory/B{number}-{self.suffix}-{file}')
   
            if len(df_list) > 0:
                try:
                    beacon_df = pd.concat(df_list).resample(f'{resample_rate}T').mean()
                except ValueError:
                    if self.verbose > 1:
                        print("\t\t\tIssue concatenating daily DataFrames")
                    # no data available so moving on to next beacon and skipping the cleaning steps
                    continue
            else:
                if self.verbose > 0:
                    print(f"\t\t\tNo data available for Beacon {beacon} between {self.start_time} - {self.end_time}")
                continue

            # Cleaning
            # --------
            # Changing NO2 readings on beacons without NO2 sensors to CO
            ##  The RPi defaults the USB directory to USB0 if only one USB port is used regardless of position
            ##  Since the CO sensor is plugged into the USB1 port and the NO2 sensor into USB0, if we don't include
            ##  the NO2 sensor then the CO sensor defaults to USB0 and the headers in the dataframe are wrong since
            ##  they are hard-coded to have USB0 correspond to NO2.
            if int(number) >= 28: # needs to be updated
                if self.verbose > 1:
                    print('\t\t\tNo NO2 sensor - switch NO2 headers to CO')

                beacon_df[['co','t_co','rh_co']] = beacon_df[['no2','t_no2','rh_no2']]
                beacon_df[['no2','t_no2','rh_no2']] = np.nan

            # converting ppb measurements to ppm
            try:
                beacon_df['co'] /= 1000
            except KeyError:
                if self.verbose > 0:
                    print("\t\t\tNo 'CO' in columns") 
            
            # getting relevant data only
            #start_date = self.beacon_id[self.beacon_id['beiwe'] == beiwe]['start_date'].values[0]
            #end_date = self.beacon_id[self.beacon_id['beiwe'] == beiwe]['end_date'].values[0]
            #beacon_df = beacon_df[start_date:end_date]

            # combining T/RH readings and dropping old columns
            try:
                beacon_df['temperature_c'] = beacon_df[['t_co','t_no2']].mean(axis=1)
                beacon_df['rh'] = beacon_df[['rh_co','rh_no2']].mean(axis=1)
                beacon_df.drop(["t_no2","t_co","rh_no2","rh_co"],axis=1,inplace=True)
            except KeyError:
                print('\t\t\tOne of ["T_NO2","T_CO","RH_NO2","RH_CO"] no found in the dataframe')

            # correcting values with linear model
            for param in self.correction.keys():
                if param not in columns_to_leave_raw:
                    if self.verbose > 1:
                        print(f"\t\t\tCorrecting {param}")
                    beacon_df[param] = beacon_df[param] * self.correction[param].loc[beacon,"coefficient"] + self.correction[param].loc[beacon,"constant"]
            
            # variables that should never have anything less than zero
            for var in ["lux",'temperature_c','rh']:
                try:
                    beacon_df[var].mask(beacon_df[var] < 0, np.nan, inplace=True)
                except KeyError:
                    if self.verbose > 1:
                        print(f"No column {var} in DataFrame")
            
            # removing extreme values 
            if extreme == 'zscore':
                # zscore greater than 2.5
                for var in beacon_df.columns:
                    beacon_df['z'] = abs(beacon_df[var] - beacon_df[var].mean()) / beacon_df[var].std(ddof=0)
                    beacon_df.loc[beacon_df['z'] > 2.5, var] = np.nan

                beacon_df.drop(['z'],axis=1,inplace=True)
            elif extreme == 'iqr':
                for var in beacon_df.columns:
                    # Computing IQR
                    Q1 = beacon_df[var].quantile(0.25)
                    Q3 = beacon_df[var].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
                    beacon_df[var].mask(beacon_df[var]<Q1-1.5*IQR,np.nan,inplace=True)
                    beacon_df[var].mask(beacon_df[var]>Q3+1.5*IQR,np.nan,inplace=True)
            else:
                if self.verbose > 1:
                    print('\t\t\tExtreme values retained')

            # removing negative values if necessary
            if retain_negative:
                if self.verbose > 1:
                    print("\t\t\tRetaining negative values")
            else:
                beacon_df[beacon_df < 0] = 0
                if self.verbose > 1:
                    print("\t\t\tRemoving negative values")

            # dropping NaN values that get in across all variables
            if retain_na:
                if self.verbose > 1:
                    print("\t\t\tRetaining NaN values")
            else:
                beacon_df.dropna(how='all',inplace=True)
                if self.verbose > 1:
                    print("\t\t\tRemoving NaN values")

            # reindexing for any missed values
            dts = pd.date_range(beacon_df.index[0],self.end_time,freq=f'1T')
            beacon_df.reindex(dts)

            # adding columns for the pt details
            beacon_df['beacon'] = beacon
            
            # combining to overall dataframe
            beacon_data = pd.concat([beacon_data,beacon_df])

        # saving
        self.beacon_data = beacon_data
        try:
            beacon_data.to_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv')
        except:
            return False

        print("DONE")
        return True

    def process_gps(self, data_dir='/Volumes/HEF_Dissertation_Research/utx000/data/raw/utx000/beiwe/gps/', home=False):
        '''
        Processes the raw gps data into one csv file for each participant and saves into /data/processed/
        
        All GPS data are recorded at 1-second intervals and stored in separate data files for every hour. The
        data are combined into one dataframe per participant, downsampled to 5-minute intervals using the
        mode value for those 5-minutes (after rounding coordinates to five decimal places), and combined into
        a final dataframe that contains all participants' data. 

        Returns True is able to process the data, false otherwise.
        '''
        print('\tProcessing gps data...')

        gps_df = pd.DataFrame()
        for participant in os.listdir(data_dir):
            if len(participant) == 8: # checking to make sure we only look for participant directories
                pid = participant
                print(f'\t\tWorking for Participant: {pid}')
                participant_df = pd.DataFrame() # 
                for file in os.listdir(f'{data_dir}{pid}/gps/'):
                    if file[-1] == 'v': # so we only import cs[v] files
                        try:
                            hourly_df = pd.read_csv(f'{data_dir}{pid}/gps/{file}',usecols=[1,2,3,4,5]) # all columns but UTC
                        except KeyError:
                            print(f'Problem with gps data for {file} for Participant {pid}')
                            self.move_to_purgatory(f'{data_dir}{pid}/gps/{file}',f'../../data/purgatory/{pid}-gps-{file}-{self.suffix}')
                    
                        if len(hourly_df) > 0: # append to participant df if there were data for that hour
                            participant_df = participant_df.append(hourly_df,ignore_index=True)
                    
                # converting utc to cdt
                participant_df['timestamp'] = pd.to_datetime(participant_df['UTC time']) - timedelta(hours=5)
                participant_df.set_index('timestamp',inplace=True)
                # rounding gps and taking the mode for every 1-minutes
                participant_df = round(participant_df,5)
                participant_df = participant_df.resample('1T').apply({lambda x: stats.mode(x)[0]})
                # converting values to numeric and removing NaN datapoints
                participant_df.columns = ['utc','lat','long','altitude','accuracy']
                for col in ['lat','long','altitude','accuracy']:
                    participant_df[col] = pd.to_numeric(participant_df[col],errors='coerce')

                participant_df.dropna(inplace=True)
                if home == True:
                    # getting participant's home coordinates
                    home_coords = self.beacon_id.set_index('beiwe')
                    home_lat = home_coords.loc[pid,'lat']
                    home_long = home_coords.loc[pid,'long']
                    # getting distance
                    R = 6.371*10**6 # radius of the earth in meters
                    participant_df['x_distance'] = abs( R * (participant_df['lat'] - home_lat) * math.pi * math.cos(home_long) / 180) 
                    participant_df['y_distance'] = abs( R * (participant_df['long'] - home_long) * math.pi / 180) 
                    dist = []
                    for i in range(len(participant_df)):
                        dist.append(math.sqrt(math.pow(participant_df.iloc[i,-2],2) + math.pow(participant_df.iloc[i,-1],2)))
                        
                    participant_df['home_distance'] = dist

                participant_df['beiwe'] = pid
                
                gps_df = gps_df.append(participant_df)

        try:
            gps_df.to_csv(f'../../data/processed/beiwe-gps-{self.suffix}.csv')
        except:
            return False

        return True

    def process_weekly_surveys(self, data_dir='../../data/raw/utx000/beiwe/survey_answers/'):
        '''
        Processes raw weekly survey answers. The survey IDs are:
        - eQ2L3J08ChlsdSXXKOoOjyLJ: morning
        - 7TaT8zapOWO0xdtONnsY8CE0: evening
        
        Returns True if able to save two dataframes for morning/evening survey data in /data/processed directory
        '''
        # defining some variables for ease of understanding
        morning_survey_id = 'eQ2L3J08ChlsdSXXKOoOjyLJ'
        evening_survey_id = '7TaT8zapOWO0xdtONnsY8CE0'
        weekly_survey_id = 'lh9veS0aSw2KfrfwSytYjxVr'
        
        # defining the final dataframes to append to
        evening_survey_df = pd.DataFrame()
        morning_survey_df = pd.DataFrame()
        weekly_survey_df = pd.DataFrame()
        
        # Morning Survey Data
        # -------------------
        print('\tProcessing morning survey data...')
        # looping through the participants and then all their data
        for participant in os.listdir(data_dir):
            # making sure we don't read from any hidden directories/files
            if len(participant) == 8:
                pid = participant
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy','TST','SOL','NAW','Restful'])
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/'):
                    # reading raw data
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/{file}')
                    # adding new row
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
                                                                                               df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with morning survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-morning-{file}-{self.suffix}')
            
                # appending participant df to overall df
                morning_survey_df = morning_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
        
        # replacing string values with numeric
        morning_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        # fixing any string inputs outside the above range
        morning_survey_df['NAW'] = pd.to_numeric(morning_survey_df['NAW'],errors='coerce')
        morning_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy','tst','sol','naw','restful']
        morning_survey_df.index.rename("timestamp",inplace=True)
        morning_survey_df = morning_survey_df.sort_index()[self.ema_start:self.ema_end]
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/'):
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-evening-{file}-{self.suffix}')
            
                evening_survey_df = evening_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        evening_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        evening_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy']
        evening_survey_df.index.rename("timestamp",inplace=True)
        evening_survey_df = evening_survey_df.sort_index()[self.ema_start:self.ema_end]

        # Weekly Survey Data
        # -------------------
        print('\tProcessing weekly survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])
            
                try:
                    for file in os.listdir(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/'):
                        df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/{file}')
                        try:
                            participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer'],df.loc[10,'answer']]
                        except KeyError:
                            try:
                                participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer']]
                            except:
                                print(f'\t\tProblem with weekly survey {file} for Participant {pid} - Participant most likely did not answer a question')
                                self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-weekly-{file}-{self.suffix}')
                
                    weekly_survey_df = weekly_survey_df.append(participant_df)
                except FileNotFoundError:
                    print(f'\t\tParticipant {pid} does not seem to have submitted any weekly surveys - check directory')
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        weekly_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Never':0,'Almost Never':1,'Sometimes':2,'Fairly Often':3,'Very Often':4,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        weekly_survey_df.columns = ['beiwe','upset','unable','stressed','confident','your_way','cope','able','top','angered','overcome']
        weekly_survey_df.index.rename("timestamp",inplace=True)
        weekly_survey_df = weekly_survey_df.sort_index()[self.ema_start:self.ema_end]

        # saving
        try:
            morning_survey_df.to_csv(f'../../data/processed/beiwe-morning_ema-{self.suffix}.csv')
            evening_survey_df.to_csv(f'../../data/processed/beiwe-evening_ema-{self.suffix}.csv')
            weekly_survey_df.to_csv(f'../../data/processed/beiwe-weekly_ema-{self.suffix}.csv')
        except:
            print("Problem saving Data")
            return False

        return True

    def process_environment_survey(self, data_file='../../data/raw/utx000/surveys/EESurvey_E1_raw.csv'):
        '''
        Processes raw environment survey (first instance) and combines relevant data into processed directory

        Returns True if processed, False otherwise
        '''
        print('\tProcessing first environment survey...')

        ee = pd.read_csv(data_file,usecols=[0,2,4,5,6,7,8,9],parse_dates=[1])
        ee.columns = ['redcap','timestamp','apartment','duplex','house','dorm','hotel','other_living']
        ee.dropna(subset=['timestamp'],inplace=True)
        ee.set_index('timestamp',inplace=True)

        # saving
        try:
            ee.to_csv(f'../../data/processed/ee-survey-{self.suffix}.csv')
        except:
            return False

        return True

    def process_fitbit(self):
        '''
        Processes fitbit data

        Returns True if processed, False otherwise
        '''
        print('\tProcessing Fitbit data...')

        def import_fitbit(filename, data_dir=f"../../data/raw/utx000/fitbit/"):
            '''
            Imports the specified file for each participant in the directory

            Inputs:
            - filename: string corresponding to the filename to look for for each participant

            Returns a dataframe with the combined data from all participants
            '''
            print(f"\tReading from file {filename}")
            df = pd.DataFrame()
            for pt in os.listdir(data_dir):
                if pt[0] != ".":
                    print(f"\t\tReading for participant {pt}")
                    try:
                        temp = pd.read_csv(f"{data_dir}{pt}/fitbit_{filename}.csv", index_col=0, parse_dates=True)
                        if filename[:4] == "intr":
                            temp = process_fitbit_intraday(temp)

                        temp["beiwe"] = pt
                        df = df.append(temp)
                    except FileNotFoundError:
                        print(f"\t\tFile {filename} not found for participant {pt}")
            df.index.rename("timestamp",inplace=True)       
            return df

        def get_device_df(info_df):
            '''
            Take dictionary-like entries for fitbit info dataframe for each row in a dataframe and makes a new dataframe
            
            Inputs:
            - info_df: the fitbit info dataframe with the dictionary-like entries
            
            Returns a dataframe for the device column
            '''
            
            overall_dict = {}
            for row in range(len(info_df)):
                Dict = ast.literal_eval(info_df['devices'][row])
                if type(Dict) == dict:
                    Dict = Dict
                elif type(Dict) in [tuple,list] and len(Dict) > 1:
                    Dict = Dict[0]
                else:
                    continue

                for key in Dict.keys():
                    overall_dict.setdefault(key, [])
                    overall_dict[key].append(Dict[key])
                # adding in the date of recording
                overall_dict.setdefault('date', [])
                overall_dict['date'].append(info_df.index[row])
                
            df = pd.DataFrame(overall_dict)
            df['timestamp'] = pd.to_datetime(df['date'],errors='coerce')
            df.drop("date",axis=1,inplace=True)
            return df.set_index('timestamp')

        def get_daily_sleep(daily_df):
            '''
            Creates a dataframe with the daily sleep data summarized
            
            Inputs:
            - daily_df: dataframe created from the daily fitbit csv file
            
            Returns a dataframe of the daily sleep data
            '''
            overall_dict = {}
            for row in range(len(daily_df)):
                # in case Fitbit didn't record sleep records for that night - value is NaN
                pt = daily_df['beiwe'][row]
                # pts with classic sleep data
                if pt in ['awa8uces','ewvz3zm1','pgvvwyvh']:
                    continue
                if type(daily_df['sleep'][row]) == float:
                    continue
                else:
                    Dict = ast.literal_eval(daily_df['sleep'][row])
                    if type(Dict) == dict:
                        Dict = Dict
                    else:
                        Dict = Dict[0]
                    for key in Dict.keys():
                        overall_dict.setdefault(key, [])
                        overall_dict[key].append(Dict[key])
                    # adding in the date of recording
                    overall_dict.setdefault('date', [])
                    overall_dict['date'].append(daily_df.index[row])
                    # adding beiwe id
                    overall_dict.setdefault('beiwe', [])
                    overall_dict['beiwe'].append(daily_df['beiwe'][row])

            df = pd.DataFrame(overall_dict)
            df['date'] = pd.to_datetime(df['date'],errors='coerce')
            # removing classic sleep stage data
            df = df[df['type'] != 'classic']
            # dropping/renaming columns
            df.drop(["dateOfSleep","infoCode","logId","type"],axis=1,inplace=True)
            df.columns = ["duration_ms","efficiency","end_time","main_sleep","levels","minutes_after_wakeup","minutes_asleep","minutes_awake","minutes_to_sleep","start_time","time_in_bed","date","beiwe"]
            df.set_index("date",inplace=True)
            return df

        def get_sleep_stages(daily_sleep):
            '''
            Creates a dataframe for the minute sleep data
            
            Input(s):
            - daily_sleep: dataframe holding the daily sleep data with a column called minuteData
            
            Returns:
            - sleep_stages: a dataframe with sleep stage data for every stage transition
            - summary: a dataframe with the nightly sleep stage information
            '''
            
            data_dict = {'startDate':[],'endDate':[],'dateTime':[],'level':[],'seconds':[],'beiwe':[]}
            summary_dict = {'start_date':[],'end_date':[],'deep_count':[],'deep_minutes':[],'light_count':[],'light_minutes':[],
                            'rem_count':[],'rem_minutes':[],'wake_count':[],'wake_minutes':[],'beiwe':[]}
            for i in range(len(daily_sleep)):
                d0 = pd.to_datetime(daily_sleep.iloc[i,:]["start_time"])
                d1 = pd.to_datetime(daily_sleep.iloc[i,:]["date"])
                sleep_dict = daily_sleep.iloc[i,:]["levels"]
                for key in sleep_dict.keys():
                    if key == 'data': # data without short wake periods
                        temp_data = sleep_dict['data']
                        for temp_data_dict in temp_data:
                            for data_key in temp_data_dict.keys():
                                data_dict[data_key].append(temp_data_dict[data_key])
                            data_dict['startDate'].append(d0.date())
                            data_dict['endDate'].append(d1.date())
                            data_dict['beiwe'].append(daily_sleep.iloc[i,:]['beiwe'])
                    elif key == 'summary': # nightly summary data - already in dictionary form
                        for summary_key in sleep_dict['summary'].keys():
                            stage_dict = sleep_dict['summary'][summary_key]
                            for stage_key in ['count','minutes']:
                                summary_dict[f'{summary_key}_{stage_key}'].append(stage_dict[stage_key])
                            
                        summary_dict['start_date'].append(d0.date())
                        summary_dict['end_date'].append(d1.date())
                        summary_dict['beiwe'].append(daily_sleep.iloc[i,:]['beiwe'])
                    else: # shortData or data with short wake periods - don't need
                        pass
                    
            sleep_stages = pd.DataFrame(data_dict)
            sleep_stages.columns = ['start_date','end_date','time','stage','time_at_stage','beiwe'] # renaming columns
            # adding column for numeric value of sleep stage 
            def numeric_from_str_sleep_stage(row):
                if row['stage'] == 'wake':
                    return 0
                elif row['stage'] == 'light':
                    return 1
                elif row['stage'] == 'deep':
                    return 2
                elif row['stage'] == 'rem':
                    return 3
                else:
                    return -1
                
            sleep_stages['value'] = sleep_stages.apply(lambda row: numeric_from_str_sleep_stage(row), axis=1)
            
            summary = pd.DataFrame(summary_dict)
            return sleep_stages, summary

        def process_fitbit_intraday(raw_df,resample_rate=1):
            '''
            Creates dataframe from the intraday measurments

            Inputs:
            - raw_df: dataframe of the raw data from Fitbit
            - resample_rate: integer specifying the minutes to downsample to

            Returns a dataframe indexed by the first column
            '''
            try:
                df = raw_df.resample(f'{resample_rate}T').mean()
            except TypeError:
                print(f"\t\tDataframe is most likely empty ({len(raw_df)})")
                return raw_df
            return df

        daily = import_fitbit("daily_records")
        daily = daily[daily['activities_steps'] > 0 ]
        
        info = import_fitbit("info")

        intra = import_fitbit("intraday_records")
        intra.columns = ["calories","steps","distance","heartrate","beiwe"]

        #device = get_device_df(info)
        print("\t\tProcessing sleep data")
        sleep_daily = get_daily_sleep(daily)
        daily.drop(['activities_heart','sleep'],axis=1,inplace=True)
        daily.columns = ["calories","bmr","steps","distance","sedentary_minutes","lightly_active_minutes","fairly_active_minutes","very_active_minutes","calories_from_activities","bmi","fat","weight","food_calories_logged","water_logged","beiwe"]
        sleep_stages, sleep_stages_summary = get_sleep_stages(sleep_daily)
        sleep_daily.drop(["levels"],axis=1,inplace=True)

        # saving
        try:
            daily.to_csv(f'../../data/processed/fitbit-daily-{self.suffix}.csv')
            info.to_csv(f'../../data/processed/fitbit-info-{self.suffix}.csv')
            intra.to_csv(f'../../data/processed/fitbit-intraday-{self.suffix}.csv')

            #device.to_csv(f'../../data/processed/bpeace2-fitbit-device.csv')
            sleep_daily.to_csv(f'../../data/processed/fitbit-sleep_daily-{self.suffix}.csv',index=False)
            sleep_stages.to_csv(f'../../data/processed/fitbit-sleep_stages-{self.suffix}.csv',index=False)
            sleep_stages_summary.to_csv(f'../../data/processed/fitbit-sleep_stages_summary-{self.suffix}.csv',index=False)
        except:
            return False

        return True
