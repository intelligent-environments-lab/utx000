# BPEACE2 Beacon Operation Check
This file analyzes the operation of the beacons during the BPEACE2 study as part of the UTx000 extension during the second half of Spring 2020 into Summer 2020.

## Beacon 1

[b1]()

### General Notes
Time between shipment and receiving is too little - RTC must have gotten damaged.

### Sensor Notes

### Debugging
 - [x] RTC: Need to check the RTC because there is no way that the beacon would have been shipped out and received/plugged in on the same day of shipping. FedEx tracking says that the beacon arrived 06/08 meaning that the battery was most likely displodged somewhere along the way. The figures shows we only had a few hours of downtime when, in reality, we should have had closer to a week. This time difference makes sense too since we should have seen the beacon logging data up until the participant moved since there is very little reason they would have unplugged it before then. The gap between the move out date and the last day of logging is just over a week which lines up pretty well.

## Beacon 5

[b5]()

## Beacon 34

[b34]()

### General Notes
The gap between sending and receiving the beacon is about 1 day which is _surprisingly_ correct for this beacon. The beacon last recorded values at 13:18 on 06/10 and was shipped out at 18:00 that day. The package was delivered at 12:01 on 06/11 according to FedEx and began recording at 13:41 on 06/12.

### Sensor Notes
This beacon has no NO2 sensor; the NO2 readings are really CO readings since the USB channels reset when only one is plugged in.

### Debugging
Dates: Need to check if the move-out date is correct or even occurred since there looks to be no break in the measurements. 

## Beacon 22

[b22]()

### General Notes
Seems the participant had their beacon plugged in up until they moved and then plugged the device in again at their new location since I did not pick the device up until 09/03. I downloaded the data from the device on the same day which is why there are no discernable data points after 09/03 corresponding to when I would have plugged in the device at my apartment. The last recorded data point by the participant was at 9:35 on 09/03 and then it seems I plugged the device in at 18:53 that day to pull the data off. 

### Sensor Notes
Seems to have been troubles with the **PM** sensor during the main study period, but the sensor seemed to be working well during the second period after the participant moved. The reverse seems to have occurred with the **CO** sensor. 

### Debugging
CO: Check if the sensor is working or just reading low in raw data.

## Beacon 24

![b24]()

### General Notes
Sensor was shipped out just after 19:00 on 06/08 (last recorded measurement at 19:04) and arrived at the participant's location on 06/10 at 14:04. Participant did not start logging data until the next day though if all the times are correct. The measurements on 06/08 after 19:00 concern me since I believe the dropbox pick up is done at 18:00 although it could be 20:00, but that seems unlikely. Also the only data collected on 06/08 is from 18:19 to 19:04 even though on 06/07 the sensor recorded up to 23:59. 

### Sensor Notes
There is a weird blip on 08/17 when all the sensors go down. Checking the data, the beacon was offline from 19:30 on 08/16 until 21:10 on 08/17. Perhaps the device was unplugged? Not sure if there is a way to find out what happened. 

### Debugging
Dates: (1) Try and determine what might have happened during the ~1 day of downtime and (2) if shipment could occur after 19:00.

## Beacon 26

![b26]()

### General Notes
Just like #24, the beacon last recorded data with me at 19:05 on 06/08 which was the day it was shipped out. It also recorded data up until 23:59 on 06/07 and then nothing on 06/08 until 18:35. Shipment arrived on 06/10 but participant did not start recording until 06/20 at 18:43 and kept recording until 8:46 on 09/08 which is the day they shipped it back. 

### Sensor Notes
GPS: We lose a good chunk of GPS data near the end of the beacon recording period most likely because the participant had their appointment with Melissa around then and deactivated their Beiwe app.

### Debugging
Dates: Can shipment could occur after 19:00?

## Beacon 46

![b46]()

### General Notes

### Sensor Notes
PM sensor seems to be exhibiting problems and no GPS data...

### Debugging
- [ ] GPS: Check again if GPS data are available

## Beacon 22

[b22]()

### Sensor Notes

### Debugging

