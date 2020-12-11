# Variable Naming Considerations and Conventions
This document outlines the naming convention used for the variety of variables listed in this study. The basic convention is discussed in addition to some exceptions to the rule.

## Root Name
For each variable, the root name will consist of as many words as necessary separated by underscores, which is typical of consistent with general [Python naming convention](https://www.crained.com/1316/best-practices-for-naming-variables-in-python/). In addtion, variable names **cannot** start with numbers or underscores, contain special characters, or consist of common programming keywords. Some examples include: 

- **BEVO Beacon:** Carbon Dioxide -> co2
- **Beiwe:** Phone Power State Level -> phone_power_level
- **Fitbit:** Daily Steps Taken -> steps_daily

Avoid using articles like "a", "the", "of", etc. when naming since they add little meaning and make variable names longer.

Acronyms can be used, but make sure they are common knowledge and not easily confused with other variables. Still keep these acronyms lowercase. For instance, good examples include:

- **BEVO Beacon:** Relative Humidity -> rh; this should be fine since if you consider the data source, the beacon, we know it measures environmental data and rh is rather self-explanatory
- **Beiwe:** Location -> gps
- **Fitbit:** Percent of Rapid Eye Movement sleep -> rem_percent

Bad examples would be the following:

- **BEVO Beacon:** Temperature -> t; should not be used since t could stand for time. Temperature can also be measured in multiple units, so something like "t_c" might be better since it avoids confusion with time and includes a unit (see below about modifiers). 
- **Beiwe:** Identifiers -> id; "id" on its own should be avoided at all times since there are many different IDs from REDCap to Beiwe. In addition, the identifiers data file from Beiwe classifies the phone, not the user. 
- **Fitbit:** Heart Rate -> hr; again someone could confuse this name with the abbreviation for hours. 

Other exceptions might arise, but use your best judgment when naming the variable.

## Variable Modifiers
In some cases, the unit associated with certain variables should be included so as to alleviate any confusion. These modifiers should be included directly **after** the root variable name and attached by an underscore. Abbreviations for units are fine so long as they are not confusing - use your best judgement when deciding. Examples include:

- **BEVO Beacon:** Temperature in Celsius -> t_c
- **Beiwe:** Phone Call Length -> phone_call_duration_seconds
- **Fitbit:** Distance Walked -> steps_distance_feet

As we begin to summarize the data, we might create derivatives or summary statistics from the raw data. These modifiers should be attached **after** the root name and any unit modifier by an underscore. Examples include:

- **BEVO Beacon:** Average CO2 -> co2_mean
- **Beiwe:** 
- **Fitbit:** Minimum Minutes Spent in REM -> rem_minutes_min

## Variable Suffix
We will keep track of the study the variable pertains to by attaching a suffix to the end of the variable name after the root and any modifiers. A Key file exists [here](https://github.com/intelligent-environments-lab/utx000/blob/master/references/WCWH%20Study%20Key.xlsx) that explains what each suffix means. An example might be:

- Median evening light level for utx000 study -> lux_evening_median_ux_s20

The suffix, "ux_s20", denotes the study title (utx000) and some indication of the start time of the study, in this case, Spring 2020. 

In the case that there were multiple deployments or samples taken, you can denote the sample/deployment number before the study title. For instance, there were two deployments of the beacons in the ut2000 study so:

- 25% percentile PM2.5 concentration from the second deployment in the UT2000 study -> pm_2p5_25percentile_2_ut2000_s19
