# Basic packages
import os

# Visual packages
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Other packages
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import re

def get_pollutant_units(pollutant):
    """Gets the formated label for the pollutant"""
    if pollutant == "co2":
        return "ppm"
    elif pollutant == "co":
        return "ppb"
    elif pollutant == "pm2p5_mass":
        return "$\mu$g/m$^3$"
    elif pollutant == "pm2p5_number":
        return "#/dL"
    elif pollutant == "no2":
        return "ppb"
    elif pollutant == "tvoc":
        return "ppb"
    elif pollutant == "temperature_c":
        return "$^\circ$C"
    elif pollutant == "rh":
        return "%"
    elif pollutant == "lux":
        return "lux"
    else:
        return ""

def get_pollutant_label(pollutant):
    """Gets the formated label for the pollutant"""
    if pollutant == "co2":
        return "CO$_2$"
    elif pollutant == "co":
        return "CO"
    elif pollutant == "pm2p5_mass" or pollutant == "pm2p5_number" or pollutant == "pm2p5p":
        return "PM$_{2.5}$"
    elif pollutant == "no2":
        return "NO$_2$"
    elif pollutant == "tvoc":
        return "TVOC"
    elif pollutant == "temperature_c":
        return "Temperature"
    elif pollutant == "rh":
        return "Relative Humidity"
    elif pollutant == "lux":
        return "Light Level"
    else:
        return ""

def get_sleep_label(metric):
    """Gets the formated label for the given sleep metric"""
    if metric[0:3].lower() == "tst":
        return "TST (h)"
    if metric.lower() == "sol":
        return "SOL (minutes)"
    if metric.lower() == "naw":
        return "NAW"
    elif metric.lower() == "efficiency":
        return "Sleep Efficiency (%)"
    elif metric.lower() == "rem_percent":
        return "% REM"
    elif metric.lower() == "nrem_percent":
        return "% nREM"
    elif metric[0:8].lower() == "rem2nrem":
        return "REM:nREM Ratio"
    else:
        return ""

class single_var:
    '''
    Visualizations that incorporate only one variable
    
    Parameters:
    - study: string specifying the study name
    '''

    def __init__(self, study_suffix):
        # figure type - for naming when saving
        self.fig_type = ''
        # study - for naming when saving
        self.study_suffix = study_suffix

    def timeseries(self, t, y, save=False, **kwargs):
        '''
        Plots a time series of the data
        
        Parameters:
        - t: list of points in time - assumed to be datetime
        - y: list of dependent values - length must be that of t
        - save: whether or not to save the figure, default is False
        
        Possible (other) Parameters:
        - figsize: tuple specifying the size of the figure - default is (16,8)
        - ylabel (or label): string specifying the ylabel - default is None
        - ylim: tuple specifying the y-axis limits - default is [min(y),max(y)]
        - yticks: list specifying the yticks to use - dfault is determined by matplotlib
        
        Returns:
        - fig: the figure handle
        - ax: the axis handle
        '''
        self.fig_type = 'timeseries'

        # setting up figure
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=(16,8))

        # plotting data
        ax.plot(t,y,linewidth=2,color='black')
        
        # Setting label
        ## x - should never be specified and will remain blank since date formats are obvious (imo)
        ax.set_xlabel('')
        ## y 
        if 'ylabel' or 'label' in kwargs.keys():
            try:
                ax.set_ylabel(kwargs['ylabel'])
            except:
                ax.set_ylabel(kwargs['label'])
        else:
            ax.set_ylabel('')
        
        # Setting limits
        ## x      
        if 'xlim' in kwargs.keys():
            ax.set_ylim(kwargs['xlim'])
        else:
            ax.set_xlim([t[0],t[-1]])
        ## y
        if 'ylim' in kwargs.keys():
            ax.set_ylim(kwargs['ylim'])
        else:
            ax.set_ylim([np.nanmin(y),np.nanmax(y)])
            
        # Setting ticks
        ## x - should never be specified
        ## xticks are determined based on the number of days included in t - breakpoints are shown below:
        ##  - dt(t) < 2 days - hourly (48 ticks)
        ##  - dt(t) < 7 weeks - daily (49 ticks)
        ##  - dt(t) < 1 year - weekly (52 ticks)
        ##  - dt(t) < 4 years - monthly (48 ticks)
        ##  - dt(t) > 4 years - yearly 
        if t[-1] - t[0] < timedelta(days = 2):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_minor_locator(mdates.HourLocator())
        elif t[-1] - t[0] < timedelta(days = 49):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%a %d'))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        elif t[-1] - t[0] < timedelta(days = 365):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        elif t[-1] - t[0] < timedelta(days = 4*365):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            
        plt.xticks(rotation=-45,ha='left')
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=-45, ha='left')
        ## y
        if 'yticks' in kwargs.keys():
            ax.set_yticks(kwargs['yticks'])
        else:
            # default
            pass
            
        # saving figure
        if save:
            # default location for Hagen's projects
            y_var = input('Shorthand for y-variable: ') ## user input for variable to identify figure
            plt.savefig(f'../../reports/figures/{y_var}-{self.fig_type}-{self.study_suffix}.png')
        
        # return the fig and axis so user can do more unique modifications
        return fig, ax

    def heatmap(self, df, col, save=False, **kwargs):
        '''
        Creates a heatmap from the data provided with days as rows and hours as columns
        
        Parameters:
        - df: dataframe indexed by datetime with a column holding the data of interest
        - col: integer or string corresponding to the column of interest in the dataframe
        
        Optional Parameters (of note):
        - colorbar: dictionary with the keys:
            - colors: colors to include in heatmap
            - ratios: must start at 0 and end at 1 and equal in length to the colors array. These values
              specify the relative locations where the true color will be
            - ticks: values to include on the colorbar - length is arbitrary

        Returns:
        - fig: the figure handle
        - ax: the axis handle
        '''
        
        self.fig_type = 'heatmap'
        # setting up figure
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=(16,8))
        
        # transforming the dataframe into the correct format
        df_by_hour = df.resample('1h').mean()
        df_by_hour['date'] = df_by_hour.index.date
        df_by_hour['hour'] = df_by_hour.index.hour
        try:
            # if column specified by integer
            df_by_hour_by_var = df_by_hour.iloc[:,[col,-1,-2]]
        except:
            # column specified by name
            df_by_hour_by_var = df_by_hour.loc[:,[col,'date','hour']]
            
        df_transformed = pd.pivot_table(df_by_hour_by_var, values=col, index='date', columns='hour')
        
        # axis ticks
        labels = []
        for d in df_transformed.index:
            labels.append(datetime.strftime(d,'%m/%d'))
            
        # colorbar
        def create_cmap(colors,nodes):
            cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
            return cmap

        # plotting
        if 'colorbar' in kwargs.keys():
            sns.heatmap(df_transformed, yticklabels=labels, vmin=kwargs['colorbar']['ticks'][0], vmax=kwargs['colorbar']['ticks'][-1],
                        cmap=create_cmap(kwargs['colorbar']['colors'],kwargs['colorbar']['ratios']),
                        cbar_kws={'ticks':kwargs['colorbar']['ticks']},
                        ax=ax)
        else:
            sns.heatmap(df_transformed, yticklabels=labels, ax=ax)
        
        # fixing axis labels
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('')
            
        # saving figure
        if save:
            # default location for Hagen's projects
            y_var = input('Shorthand for y-variable: ') ## user input for variable to identify figure
            plt.savefig(f'../../reports/figures/{y_var}-{self.fig_type}-{self.study_suffix}.png')
        
        # return the fig and axis so user can do more unique modifications
        return fig, ax

class Beacon_Visual():

    def __init__(self, beacon, start_datetime, end_datetime, data_dir="../../data/processed", study="ux_s20"):
        self.beacon = beacon
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.data_dir = data_dir
        self.study = study

        self.data = pd.read_csv(f"{data_dir}/beacon-{self.study}.csv",index_col=0,parse_dates=True,infer_datetime_format=True)
        self.data = self.data[self.start_datetime.strftime("%m-%d-%Y %H:%M"):self.end_datetime.strftime("%m-%d-%Y %H:%M")]

    def timeseries(self, variable):
        """
        Plots a basic time series and saves it to the user's desktop
        """
        _, ax = plt.subplots(figsize=(12,6))
        ax.plot(self.data.index,self.data[variable])
        ax.set_title(variable.upper())
        plt.savefig(f"{self.data_dir}/../../reports/inspection/{variable.lower()}-timeseries-{self.start_datetime.date()}-{self.end_datetime.date()}-{self.study}.png")
        plt.close()

def main():
    os.system("clear")

    print("Please choose an option from the list below")
    print("\t1. Beacon Visualization")
    op = int(input("\nOption: "))

    if op == 1:
        print("Please enter the following information:")
        bb = int(input("\tBeacon: "))
        var = (input("\tVariable: "))
        start_date = input("\tStarting date (defaults to today): ")
        start_time = input("\tStarting time (hour and minute only - defaults to 00:00): ")
        end_date = input("\tEnding date (defaults to today): ")
        end_time = input("\tEnding time (hour and minute only - defaults to 23:59): ")

        # defaults
        if start_date == "":
            start_date = datetime.now().date().strftime("%m/%d/%Y")

        if start_time == "":
            start_time = "0:0"

        if end_date == "":
            end_date = datetime.now().date().strftime("%m/%d/%Y")

        if end_time == "":
            end_time = "23:59"

        d = re.split("/|,|-",start_date)
        t = re.split("/|,|-|:",start_time)
        start_datetime = datetime(int(d[2]),int(d[0]),int(d[1]),int(t[0]),int(t[1])) # assuming m/d/Y H:M format
        d = re.split("/|,|-",end_date)
        t = re.split("/|,|-|:",end_time)
        end_datetime = datetime(int(d[2]),int(d[0]),int(d[1]),int(t[0]),int(t[1])) # assuming m/d/Y H:M format

        bv = Beacon_Visual(bb, start_datetime, end_datetime)

        print("\nPlease choose the visualization type from the list below")
        print("\t1. Time Series")
        op = int(input("\nOption: "))
        if op == 1:
            bv.timeseries(var)
    else:
        print("Not a valid choice - please run again")

if __name__ == '__main__':
    main()
