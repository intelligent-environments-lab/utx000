# Using information fusion across sationary, mobile, and wearable consumer-grade sensors to confidently estimate bedroom ventilation rates

_One of these days I will come up with a succinct title_

**Main Idea**: Calculate ventilation rates in UTx000 participants' bedrooms using steady-state and decay methods. While this process is not novel, the _more_ interesting part comes from the data fusion component. I showcase the methods I use to confidently determine when bedrooms are occupied:

1.  **Cross-referencing GPS and Fitbit-detected sleep events**: GPS confirms that participants are home and Fitbit confirms when participants are asleep i.e. in their bedrooms (assuming people don't pass out on the couch)
2. **Occupancy Detection**: Using the dataset from (1) and a dataset that corss-references GPS data when participants are _not_ home with IAQ data, I can label my IAQ data as either being from _occupied_ or _unoccupied_ periods. Then I train a model (MLP seems to work best) per participant on the CO2 and TVOC parameters and can predict, with 90% or more confidence, occupied bedroom periods that we don't get from (1) because GPS is down or participants didn't wear their Fitbit to bed. This process helps augment the dataset from (1) and in some cases provides 4000+ extra IAQ data points (at a 10-minute resolution).

I can also separate out the ventilation rates estimated from (1) vs (2) which could indicate the utility of the occupancy-detection-based dataset. 

**Main Contribution**: I would say this study highlights the ability of consumer-grade sensors to provide powerful/useful information to curate _accurate_ datasets for other analyses. So the main idea might be less about calculating ventilation rates and more about the method, but not sure how best to frame that. 

## Analysis Required

This work is a combination of the work I have done for ASHRAE and Indoor Air in combinarion with some of the lesser work for the WCWH Showcase and GAIN events. Thus most of the work is done. I would need to calculate ventilation rates using data from (2) which will require a bit of processing to get (2) into a good form. Once processed, I would just need to run (2) through the script I already developed to calculate ventilation rates. 