# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:56:23 2021

@author: linca
"""
import time
import os
import numpy as np
import pandas as pd

class Beacon:
    def __init__(self, path):
        self.path = path
        self.number = path[path.rfind('/')+2:].lstrip('0')
        self.filepaths = {'adafruit':[f'{self.path}/adafruit/{file}' for file in os.listdir(f'{self.path}/adafruit')],
                      'sensirion':[f'{self.path}/sensirion/{file}' for file in os.listdir(f'{self.path}/sensirion')]}
        
        self.columns = {'adafruit':['Timestamp', 'TVOC', 'eCO2', 'Lux', 'Visible', 'Infrared', 'NO2',
                                    'T_NO2', 'RH_NO2', 'CO', 'T_CO', 'RH_CO'],
                        'sensirion':['Timestamp','Temperature [C]','Relative Humidity','CO2','PM_N_0p5','PM_N_1','PM_N_2p5','PM_N_4','PM_N_10','PM_C_1','PM_C_2p5','PM_C_4','PM_C_10']
                        }
    
    def read_csv(self):  
        def _read_csv(path):
            print(self.number+"csv")
            columns=None
            if 'adafruit' in path:
                columns=self.columns['adafruit']
            elif 'sensirion' in path:
                columns=self.columns['sensirion']
            try:
                return pd.read_csv(path, index_col='Timestamp',usecols=columns,parse_dates=True,infer_datetime_format=True)
            except ValueError:
                return pd.DataFrame()
            
        self.sensirion = pd.concat((_read_csv(file) for file in self.filepaths['sensirion']), copy=False).resample('5T').mean()
        # print("don one")
        self.adafruit = pd.concat((_read_csv(file) for file in self.filepaths['adafruit']), copy=False).resample('5T').mean()
    
    def preprocess(self):
        number = self.number
        adafruit = self.adafruit
        sensirion = self.sensirion
        
        def mislabeled_NO2(df):
            # Mis-wiring NO2 sensor doesn't actually exist
            df[['CO','T_CO','RH_CO']] = df[['NO2','T_NO2','RH_NO2']]
            df[['NO2','T_NO2','RH_NO2']] = np.nan
            return df

        if number in [28,29]:
            adafruit = mislabeled_NO2(adafruit)
        adafruit['CO'] /= 1000 #ppb to ppm
        
        beacon_df = adafruit.merge(right=sensirion,left_index=True,right_index=True,how='outer')

        def nonnegative(df):
            for var in ['CO2','T_NO2','T_CO','Temperature [C]','RH_NO2','RH_CO','Relative Humidity']:
                df[var].mask(df[var] < 0, np.nan, inplace=True)
            return df
        
        def lower_bound(df):
            for var, threshold in zip(['CO2','Lux'],[100,-1]):
                df[var].mask(df[var] < threshold, np.nan, inplace=True)
            return df
            
        beacon_df = lower_bound(nonnegative(beacon_df))
        beacon_df['Beacon'] = self.number
        beacon_df = beacon_df.reset_index().set_index(['Beacon','Timestamp'])
        self.data = beacon_df
    
    @property
    def empty(self):
        return len(self.filepaths['adafruit']+self.filepaths['sensirion'])<1
    
    def __str__(self):
        return f'Beacon object at {self.path}'
    
    def __repr__(self):
        return f'Beacon object at {self.path}'
        
    
class BPeace:
    def __init__(self, beacon_list):
        self.beacons_folder = '../../data/raw/utx000/beacon'
        self.beacon_list = np.sort(beacon_list).tolist()
        
    def process_beacon(self):
        beacons = [Beacon(f'{self.beacons_folder}/B{beacon:02}') for beacon in self.beacon_list]
        beacons = [beacon for beacon in beacons if not beacon.empty]
        start = time.perf_counter()
        for beacon in beacons:
            beacon.read_csv()

        print(f'{time.perf_counter()-start} seconds')
        start = time.perf_counter()
        for beacon in beacons:
            beacon.preprocess()
        print(f'{time.perf_counter()-start} seconds')
        start = time.perf_counter()
        self.beacon_data = pd.concat([beacon.data for beacon in beacons])
        print(f'{time.perf_counter()-start} seconds')        
        self.beacons=beacons
        # TODO:  give real filename to this
        start = time.perf_counter()
        self.beacon_data.to_parquet('..\..\data\interim\utx000_beacon.parquet')
        print(f'{time.perf_counter()-start} seconds')  

if __name__=='__main__':
    beacon_list = [1,5,6,7,10,11,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,38,40,41,44,46,48]
    bpeace = BPeace(beacon_list)
    
    start = time.perf_counter()
    bpeace.process_beacon()
    end = time.perf_counter()
    print(f'{end-start} seconds')
    
