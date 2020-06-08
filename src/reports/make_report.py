# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime

import logging
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import jinja2
import pdfkit

class beiwe_participation_report():
    '''

    '''
    def __init__(self,report_date):
        '''

        '''

        self.report_date = report_date

    def load_survey_data(self,dateThru='06082020'):
        '''

        '''

        # importing
        self.morn = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/interim/survey_mood_morning_summary_thru{dateThru}.csv',
                  index_col=0)
        self.night = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/interim/survey_mood_evening_summary_thru{dateThru}.csv',
                  index_col=0)
        self.week = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/interim/survey_mood_week_summary_thru{dateThru}.csv',
                  index_col=0)

        # cleaning
        for df in [self.morn,self.night,self.week]:
            
            for column in df.columns:
                df[column] = pd.to_numeric(df[column],errors='coerce')
                
            if 'sum' in df.columns:
                pass
            else:
                df['sum'] = df.sum(axis=1)

    def load_sensor_data(self,dateThru='06082020'):
        '''

        '''
        self.acc = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/interim/acc_summary_thru{dateThru}.csv',
                  index_col=0)

    def create_plots(self):
        '''

        '''
        # morning and evening histograms
        fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
        i = 0
        colors = ('firebrick','cornflowerblue')
        for df,name in zip([self.morn,self.night],['morning','evening']):
            ax = axes[i]
            sns.distplot(df['sum'],bins=np.arange(0,16,1),color=colors[i],kde=False,label=name,ax=ax)
            ax.set_xticks(np.arange(0,16,1))
            ax.set_xlim([0,15])
            ax.set_ylim([0,12])
            ax.set_xlabel('Total Surveys Completed')
            ax.legend(title='Survey Timing')
            i += 1

        axes[0].set_ylabel('Number of Participants')
        axes[1].set_xticks(np.arange(1,16))
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
        plt.savefig('/Users/hagenfritz/Projects/utx000/reports/beiwe_check/daily_survey_summary_histogram.png')

        # weekly histograms
        fig,ax = plt.subplots(figsize=(12,6))
        sns.distplot(self.week['sum'],bins=np.arange(0,6,1),kde=False,color='cornflowerblue')
        ax.set_xticks(np.arange(0,6,1))
        ax.set_xlabel('Total Surveys Completed')
        ax.set_xlim([0,5])
        ax.set_ylabel('Number of Participants')

        plt.savefig('/Users/hagenfritz/Projects/utx000/reports/beiwe_check/weekly_survey_summary_histrogram.png')

        # moring and evening time series
        fig,ax = plt.subplots(figsize=(12,6))
        for df,name in zip([self.morn,self.night],['morning','evening']):
            dates = []
            daily = []
            for column in df.columns:
                if column == 'sum':
                    continue
                    
                dates.append(datetime.strptime(column,'%m/%d/%Y'))
                daily.append(np.sum(df[column]))
                
            #ax.stem(dates,daily)
            if name == 'morning':
                plt.vlines(x=dates, ymin=0, ymax=daily, color='orange',alpha=1)
                plt.scatter(dates,daily,s=50, color='orange',edgecolor='black',alpha=1,label='morning')
            else:
                plt.vlines(x=dates, ymin=0, ymax=daily, color='purple',alpha=1)
                plt.scatter(dates,daily,s=100, color='purple',edgecolor='black',alpha=1,label='evening')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            plt.xticks(rotation=-30,ha='left')
            ax.set_ylabel('Number of Surveys Submitted Daily')
            ax.set_ylim([0,30])
            ax.set_yticks(np.arange(0,31,2))

        ax.grid(axis='y')    
        ax.legend(title='Survey Timing')

        plt.savefig('/Users/hagenfritz/Projects/utx000/reports/beiwe_check/daily_survey_timeseries.png')

        # weekly time series
        fig,ax = plt.subplots(figsize=(12,6))
        df = self.week
        dates = []
        daily = []
        for column in df.columns:
            if column == 'sum':
                continue

            dates.append(datetime.strptime(column,'%m/%d/%Y'))
            daily.append(np.sum(df[column]))

        plt.vlines(x=dates, ymin=0, ymax=daily, color='orange',alpha=1)
        plt.scatter(dates,daily,s=50, color='orange',edgecolor='black',alpha=1,label='morning')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=-30,ha='left')
        ax.set_ylabel('Number of Surveys Submitted Daily')
        ax.set_ylim([0,30])
        ax.set_yticks(np.arange(0,31,2))

        ax.grid(axis='y')
        plt.savefig('/Users/hagenfritz/Projects/utx000/reports/beiwe_check/weekly_survey_timeseries.png')

        # acc time series
        fig,ax = plt.subplots(figsize=(12,6))
        df = self.acc
        dates = []
        daily = []
        for column in df.columns:
            if column == 'sum':
                continue

            dates.append(datetime.strptime(column,'%m/%d/%y'))
            daily.append(np.sum(df[column]))

        #ax.plot(dates,daily)
        plt.vlines(x=dates, ymin=0, ymax=daily, color='orange',alpha=1)
        plt.scatter(dates,daily,s=50, color='orange',edgecolor='black',alpha=1,label='morning')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=-30,ha='left')
        ax.set_xlim([datetime(2020,4,24),datetime(2020,5,31)])
        ax.set_ylim([0,1750000])
        ax.set_ylabel('Number of Daily Bytes')

        plt.savefig('/Users/hagenfritz/Projects/utx000/reports/beiwe_check/acc_timeseries.png')

    def get_filename(self,filename):

        return f'/Users/hagenfritz/Projects/utx000/reports/beiwe_check/{filename}.png'

    def generate_report(self):
        '''

        '''

        templateLoader = jinja2.FileSystemLoader(searchpath="/Users/hagenfritz/Projects/utx000/reports/templates/")
        templateEnv = jinja2.Environment(loader=templateLoader)
        templateEnv.globals['get_filename'] = self.get_filename
        TEMPLATE_FILE = "beiwe_participation_template.html"
        template = templateEnv.get_template(TEMPLATE_FILE)
        outputText = template.render(date=self.report_date)
        html_file = open(f'/Users/hagenfritz/Projects/utx000/reports/beiwe_check/report_{self.report_date}.html', 'w')
        html_file.write(outputText)
        html_file.close()

    def generate_report_from_interim(self):
        '''

        '''
        self.load_survey_data(self.report_date)
        self.load_sensor_data(self.report_date)
        self.create_plots()
        self.generate_report()

def main():
    '''
    Generates reports
    '''
    logger = logging.getLogger(__name__)

    # Generate Biewe Check Report
    logger.info('Generating Beiwe Participation Check Report...')
    dateThru = input('Date Thru for Beiwe Participation Check (%m%d%Y): ')
    report_gen = beiwe_participation_report(dateThru)
    report_gen.generate_report_from_interim()
    logger.info('Generated')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='report.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()