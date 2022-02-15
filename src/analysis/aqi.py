import pandas as pd
import numpy as np

def co2(value,qualitative=False):
    """
    AQI for CO2 based on: https://www.breeze-technologies.de/blog/calculating-an-actionable-indoor-air-quality-index/

    Parameters
    ----------
    value : float
        measured co2 concentration in ppm
    qualitative : boolean, default False
        whether to return AQI as a qualitative label

    Returns
    -------
    aqi : int or str
        quantitative or qualitative summary of co2 measurement
    """
    mapping = {1:"excellent",2:"fine",3:"moderate",4:"poor",5:"very_poor",6:"severe"}

    if value == np.nan:
        return np.nan

    if value <= 400:
        aqi = 1
    elif value <= 1000:
        aqi = 2
    elif value <= 1500:
        aqi = 3
    elif value <= 2000:
        aqi = 4
    elif value <= 5000:
        aqi = 5
    else:
        aqi = 6

    if qualitative:
        return mapping[aqi]
    else:
        return aqi

def tvoc(value,qualitative=False):
    """
    AQI for TVOC measurements based on: https://www.breeze-technologies.de/blog/calculating-an-actionable-indoor-air-quality-index/

    Parameters
    ----------
    value : float
        measured tvoc concentration in ppb
    qualitative : boolean, default False
        whether to return AQI as a qualitative label

    Returns
    -------
    aqi : int or str
        quantitative or qualitative summary of tvoc measurement
    """
    mapping = {1:"excellent",2:"fine",3:"moderate",4:"poor",5:"very_poor",6:"severe"}

    if value == np.nan:
        return np.nan

    if value <= 50:
        aqi = 1
    elif value <= 100:
        aqi = 2
    elif value <= 150:
        aqi = 3
    elif value <= 200:
        aqi = 4
    elif value <= 300:
        aqi = 5
    else:
        aqi = 6

    if qualitative:
        return mapping[aqi]
    else:
        return aqi

def pm2p5_mass(value,indoor=False,qualitative=False):
    """
    AQI for PM2.5 based on: https://www.airnow.gov/aqi/aqi-basics/

    Parameters
    ----------
    value : float
        measured pm2p5 concentration in ug/m3
    indoor : boolean, default False
        whether to scale aqi thresholds based on indoor or outdoor measurements. Indoor scale is half of outdoor scale. 
    qualitative : boolean, default False
        whether to return AQI as a qualitative label

    Returns
    -------
    aqi : int or str
        quantitative or qualitative summary of pm2p5 measurement
    """
    mapping = {1:"excellent",2:"fine",3:"moderate",4:"poor",5:"very_poor",6:"severe"}
    if indoor:
        factor = 2
    else:
        factor = 1

    if value == np.nan:
        return np.nan

    if value <= 50/factor:
        aqi = 1
    elif value <= 100/factor:
        aqi = 2
    elif value <= 150/factor:
        aqi = 3
    elif value <= 200/factor:
        aqi = 4
    elif value <= 300/factor:
        aqi = 5
    else:
        aqi = 6
    
    if qualitative:
        return mapping[aqi]
    else:
        return aqi

def trh(t,rh,qualitative=False,data_dir="../../data/"):
    """
    AQI for T/RH based on: https://www.breeze-technologies.de/blog/calculating-an-actionable-indoor-air-quality-index/

    Parameters
    ----------
    t : float
        measured temperature in C
    rh : float
        measured relative humidity as percent
    qualitative : boolean, default False
        whether to return AQI as a qualitative label

    Returns
    -------
    aqi : int or str
        quantitative or qualitative summary of trh measurements
    """
    mapping = {1:"excellent",2:"fine",3:"moderate",4:"poor",5:"very_poor",6:"severe"}
    trh_table = pd.read_csv(f"{data_dir}/external/trh_aqi_table.csv",index_col=0)

    if t == np.nan:
        return np.nan
    if rh == np.nan:
        return np.nan

    try:
        aqi = trh_table.loc[round(rh,1),str(int(t))]
    except KeyError:
        aqi = 6
    except ValueError:
        return np.nan

    if qualitative:
        return mapping[aqi]
    else:
        return aqi

