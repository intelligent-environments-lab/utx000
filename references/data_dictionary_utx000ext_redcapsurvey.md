# UTx000 Extension Survey "Data Dictionary"
This file breaks down the various surveys administered via UT's RedCAP system. While not a true data dictionary, the various fields are contained here. The names of the following sections are taken directly from those sent out via RedCAP for easy cross-referencing.

## Consent Form

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 1 |
| Event Name | Str | NA | Event 1 |
| Survey Timestamp | Str | NA | 11/10/19 16:53 | 
| I DO agree to wear a FitBit activity monitor, install a smartphone application and respond to weekly queries throughout the study period. | Int | 0, 1 | 1 |
| I DO agree that my data can be linked with other data being collected by the University of Texas including administrative data (e.g. number of hours completed, year of high school graduation, high school GPA) and used by the study investigators for research purposes | Int | 0, 1 | 1 |
| I DO agree that I may be re-contacted for future study opportunities. | Int | 0, 1 | 1 |
| I DO agree to complete online surveys assessing variables such as food intake, lifestyle/behavior, social factors, demographics, socioeconomic status, psychological factors, health, medication intake, cognition, home environment, life events, hair care/length, study-related feedback, and my activities and moods during the COVID-19 quarantine period. | Int | 0, 1 | 1 |
| Complete? | Str | Complete, Incomplete | Complete |

## Beiwe Instructions
While there is a label for this "survey", this was more of an informational email about how to set up and register the Beiwe application. The survey was meant as a way to check that a participant had acknowledged the email. 

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 4 |
| Event Name | Str | NA | Event 1 |
| Survey Timestamp | Str | NA | 7/11/19 15:28 | 
| Complete? | Str | Complete, Incomplete | Complete |

## Survey Instructions
Similar to above, this "survey" was just meant to check that a participant had read and received the notification. 

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 9 |
| Event Name | Str | NA | Event 1 |
| Survey Timestamp | Str | NA | 04/30/20 15:54 | 
| Complete? | Str | Complete, Incomplete | Complete |

## Environment and Experiences Questionnaire

* Fields were repeated multiple times and answered based on number of roommates

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 1 |
| Event Name | Str | NA | Event 1 |
| Survey Timestamp | Str | NA | 11/10/19 16:58 |
| In the past week, have you traveled to a different city? | Str | Yes, No | No |
| What type of building are you currently living in? (choice=Apartment) | Str | Unchecked, Checked | Checked |
| What type of building are you currently living in? (choice=Duplex) | Str | Unchecked, Checked | Checked |
| What type of building are you currently living in? (choice=Stand-alone House) | Str | Unchecked, Checked | Checked |
| What type of building are you currently living in? (choice=Dormitory) | Str | Unchecked, Checked | Checked |
| What type of building are you currently living in? (choice=Motel/Hotel) | Str | Unchecked, Checked | Checked |
| What type of building are you currently living in? (choice=Other) | Str | Unchecked, Checked | Checked |
| Are you living with the same people as last week? | Str | Yes, No | Yes |
| Gender (choice=M)* | Str | Unchecked, Checked | Checked |
| Gender (choice=F)* | Str | Unchecked, Checked | Checked |
| Gender (choice=Nonbinary)* | Str | Unchecked, Checked | Checked |
| Family Member* | Str | Yes, No | No |
| In the past week, have you opened the windows in your residence? | Str | Yes, No | No |
| Why did you open your windows? (choice=To change indoor temperature) | Str | Unchecked, Checked | Checked |
| Why did you open your windows? (choice=To get fresh air) | Str | Unchecked, Checked | Checked |
| Why did you open your windows? (choice=To change temperature and get fresh air) | Str | Unchecked, Checked | Checked |
| Why did you open your windows? (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, have you tried to isolate yourself in some part of your home? | Str | Yes, No | Yes |
| Why? (choice=To be able to work without interruption) | Str | Unchecked, Checked | Checked |
| Why? (choice=To have more privacy) | Str | Unchecked, Checked | Checked |
| Why? (choice=To isolate from a sick member of the household) | Str | Unchecked, Checked | Checked |
| Why? (choice=To isolate myself because I am not feeling well) | Str | Unchecked, Checked | Checked |
| Why? (choice=Other) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Stayed in bedroom) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Stayed in another room of the house) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Closed windows in room) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Tried to change ventilation in the isolated space) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Minimized contact with other household members) | Str | Unchecked, Checked | Checked |
| How did you try to isolate yourself? (choice=Other) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Stagnant) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Smelly) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Earthy) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Moldy) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Cooking) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Fragrant) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Fresh (well ventilated)) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Obnoxious) | Str | Unchecked, Checked | Checked |
| Based on your sense of smell, how can you describe the 'freshness' of air at your residence? (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, have you changed your air conditioning filter? | Str | Yes, No | No |
| Are you currently using a portable air cleaner at home? | Str | Yes, No | Yes |
| Did you use the air cleaner prior to the COVID-19 outbreak? | Str | Yes, No | Yes |
| In the past week, has your house been uncomfortably hot? | Str | Yes, No | Yes |
| Can you easily control the temperature within your home? | Str | Yes, No | Yes |
| In the past week, has your home been uncomfortably humid (sticky)? | Str | Yes, No | Yes |
| In the past week, have the hard surfaces in your residence been cleaned? | Str | Yes, No | Yes |
| What hard surfaces were cleaned? (choice=Kitchen Counters) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Floors) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Door Knobs) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Table Tops) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Sinks) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Toilets) | Str | Unchecked, Checked | Checked |
| What hard surfaces were cleaned? (choice=Appliance Knobs and Handles) | Str | Unchecked, Checked | Checked |
| In the past week, has your home been vacuumed? | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Bleach) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Ammonia) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Pinesol) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Vinegar) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Alcohol) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Disinfectant wipes) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Soap and Water) | Str | Unchecked, Checked | Checked |
| In the past week, have any of the following cleaners been used in your home (Check all that apply )? (choice=Floor Cleaners) | Str | Unchecked, Checked | Checked |
| Bleach: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Bleach: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Bleach: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Bleach: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Bleach: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Ammonia: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Ammonia: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Ammonia: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Ammonia: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Ammonia: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Pinesol: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Pinesol: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Pinesol: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Pinesol: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Pinesol: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Vinegar: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Vinegar: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Vinegar: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Vinegar: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Vinegar: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Alcohol: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Alcohol: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Alcohol: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Alcohol: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Alcohol: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Disinfectant wipes: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Disinfectant wipes: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Disinfectant wipes: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Disinfectant wipes: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Disinfectant wipes: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Soap and water: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Soap and water: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Soap and water: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Soap and water: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Soap and water: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| Floor Cleaners: In which rooms? (choice=Kitchen) | Str | Unchecked, Checked | Checked |
| Floor Cleaners: In which rooms? (choice=Bathroom) | Str | Unchecked, Checked | Checked |
| Floor Cleaners: In which rooms? (choice=Living Room) | Str | Unchecked, Checked | Checked |
| Floor Cleaners: In which rooms? (choice=Bedroom) | Str | Unchecked, Checked | Checked |
| Floor Cleaners: In which rooms? (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, have you changed your home cleaning practices in response to the COVID-19 outbreak?| Str | Yes, No | No | 
| How? (choice=Cleaning more frequently) | Str | Unchecked, Checked | Checked |
| How? (choice=Cleaning different surfaces) | Str | Unchecked, Checked | Checked |
| How? (choice=Using different cleaners) | Str | Unchecked, Checked | Checked |
| How? (choice=Other) | Str | Unchecked, Checked | Checked |
| What do you most often use to clean your hands with? (choice=Soap) | Str | Unchecked, Checked | Checked |
| What do you most often use to clean your hands with? (choice=Alcohol Based Hand Sanitizer) | Str | Unchecked, Checked | Checked |
| What do you most often use to clean your hands with? (choice=Other disinfectant) | Str | Unchecked, Checked | Checked |
| What do you most often use to clean your hands with? (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, have you changed your hand cleaning practices in response to the COVID-19 outbreak? | Str | Unchecked, Checked | Checked |
| How? (choice=Clean more frequently) | Str | Unchecked, Checked | Checked |
| How? (choice=Clean for longer) | Str | Unchecked, Checked | Checked |
| How? (choice=Use different hand cleaners) | Str | Unchecked, Checked | Checked |
| Immediately when returning to your home did you: (choice=Leave your street shoes by the door or outside?) | Str | Unchecked, Checked | Checked |
| Immediately when returning to your home did you: (choice=Wash the street clothes you were wearing?) | Str | Unchecked, Checked | Checked |
| Immediately when returning to your home did you: (choice=Shower?) | Str | Unchecked, Checked | Checked |
| If you bought groceries and/or home supplies, did you clean the packaging? | Str | Unchecked, Checked | Checked |
| Which product/technique did you use? (choice=Soap) | Str | Unchecked, Checked | Checked |
| Which product/technique did you use? (choice=Disinfectant wipes) | Str | Unchecked, Checked | Checked |
| Which product/technique did you use? (choice=Alcohol-based Solution) | Str | Unchecked, Checked | Checked |
| Which product/technique did you use? (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=None) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=Less than 2) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=Less than 5) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=Less than 10) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=Less than or equal 20) | Str | Unchecked, Checked | Checked |
| In the past week, approximately how many people have you directly interacted with in person (excluding those you live with)? (choice=Less than or equal 20) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=None) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=Less than 2) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=Less than 5) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=Less than 10) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=Less than or equal 20) | Str | Unchecked, Checked | Checked |
| In the past week, how often did your housemates leave the house? (choice=Greater than 20) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Participated in on-line UT Austin courses, discussions or activities) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Cooked dinner at home?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Picked up take-out food) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Had take-out food delivered to your home?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Had groceries delivered to your home?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone into a grocery store?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone into a pharmacy?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone into a doctors office) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone into a hospital?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone to work outside the home) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone for a walk outside) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone outside to workout) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone to a park?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Gone to a City trail to walk or run?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Exercised at home?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Left your home to run an errand?) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Driven in a car by yourself) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Driven in a car with someone else) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Taken a bus) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Taken a rideshare) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Taken a flight) | Str | Unchecked, Checked | Checked |
| In the previous week, have you: (choice=Made facecalls (facetime, whatsapp, zoom)  with Family or Friends) | Str | Unchecked, Checked | Checked |
| In the past week, have you changed your plans to limit contact with people outside your household? | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Not leaving your home) | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Limiting visitors to home) | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Other) | Str | Unchecked, Checked | Checked |
| In the past week, have you suffered from seasonal allergy symptoms? | Str | Yes, No | Yes |
| Season allergy symptoms are  (choice=Significantly worse than last week) | Str | Unchecked, Checked | Checked |
| Season allergy symptoms are  (choice=Somewhat worse than last week) | Str | Unchecked, Checked | Checked |
| Season allergy symptoms are  (choice=Same as last week) | Str | Unchecked, Checked | Checked |
| Season allergy symptoms are  (choice=Somewhat better than last week) | Str | Unchecked, Checked | Checked |
| Season allergy symptoms are  (choice=Significantly better than last week) | Str | Unchecked, Checked | Checked |
| Have you ever received an asthma diagnosis from a doctor? | Str | Yes, No | No |
| In the past week, have you exhibited COVID-19 like symptoms? | Str | Yes, No | No |
| Check all that apply (choice=Cough) | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Fever) | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Tiredness) | Str | Unchecked, Checked | Checked |
| Check all that apply (choice=Shortness of breath) | Str | Unchecked, Checked | Checked |
| In the past week, have you received a positive COVID-19 test result? | Str | Yes, No | No |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Never used masks or gloves) | Str | Unchecked, Checked | Checked |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Wanted to use masks or gloves but they were unavailable to me (e.g. sold out)) | Str | Unchecked, Checked | Checked |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Occasionally wore masks) | Str | Unchecked, Checked | Checked |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Occasionally wore gloves) | Str | Unchecked, Checked | Checked |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Always used masks) | Str | Unchecked, Checked | Checked |
| In the past week, have you used any personal protective equipment such as gloves or masks when going outside or meeting with others? (choice=Always used gloves) | Str | Unchecked, Checked | Checked |
| Complete? | Str | Incomplete, Complete | Complete |

## Spring Break Questions Only Once

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 12 |
| Event Name | Str | NA | Event 3 |
| Timestamp | Str | NA | 1/1/20 16:38 |
| Did you travel out of Austin during the last week of classes (March 9-13th) | Str | Yes, No | No |
| Reasons for travel (choice=Vacation) | Str | Unchecked, Checked | Unchecked |
| Reasons for travel (choice=Return home) | Str | Unchecked, Checked | Unchecked |
| Reasons for travel (choice=Work) | Str | Unchecked, Checked | Unchecked |
| Reasons for travel (choice=Other) | Str | Unchecked, Checked | Unchecked |
| How did you travel (check all that apply)? (choice=Car) | Str | Unchecked, Checked | Unchecked |
| How did you travel (check all that apply)? (choice=Bus) | Str | Unchecked, Checked | Unchecked |
| How did you travel (check all that apply)? (choice=Plane) | Str | Unchecked, Checked | Unchecked |
| How did you travel (check all that apply)? (choice=Rideshare) | Str | Unchecked, Checked | Unchecked |
| How did you travel (check all that apply)? (choice=Other) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Cooked dinner at home?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Picked up take-out food) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Had take-out food delivered to your home?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Had groceries delivered to your home?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a grocery store?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a pharmacy?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a doctors office) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a hospital?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to work outside the home) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone for a walk outside) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone outside to workout) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a park?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a City trail to walk or run?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a lake, river or stream to swim?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a beach?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a gym?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Exercised at home?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Eaten in a restaurant?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone into a bar) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a gathering or meeting of 10 or more people) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Gone to a movie) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Played board or card games in person with others?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Played a team sport?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Hosted visitors in your home) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Left your home to run an errand?) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Driven in a car by yourself) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Driven in a car with someone else) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Taken a bus) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Taken a rideshare) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Taken a flight) | Str | Unchecked, Checked | Unchecked |
| During the last week of classes, what activities did you participate in: (choice=Made facecalls (facetime, whatsapp, zoom)  with Family or Friends) | Str | Unchecked, Checked | Unchecked |
| Did you travel out of Austin during the first week of spring break (March 14-21st)? | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (Check all that apply) (choice=Vacation) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (Check all that apply) (choice=Return home) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (Check all that apply) (choice=Work) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (Check all that apply) (choice=Other) | Str | Unchecked, Checked | Unchecked |
| If Y, how did you travel? (check all that apply) (choice=Car) | Str | Unchecked, Checked | Unchecked |
| If Y, how did you travel? (check all that apply) (choice=Bus) | Str | Unchecked, Checked | Unchecked |
| If Y, how did you travel? (check all that apply) (choice=Plane) | Str | Unchecked, Checked | Unchecked |
| If Y, how did you travel? (check all that apply) (choice=Rideshare) | Str | Unchecked, Checked | Unchecked |
| If Y, how did you travel? (check all that apply) (choice=Other) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Cooked dinner at home?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Picked up take-out food) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Had take-out food delivered to your home?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Had groceries delivered to your home?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a grocery store?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a pharmacy?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a doctors office) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a hospital?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to work outside the home) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone for a walk outside) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone outside to workout) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a park?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a City trail to walk or run?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a lake, river or stream to swim?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a beach?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a gym?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Exercised at home?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Eaten in a restaurant?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone into a bar) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a gathering or meeting of 10 or more people) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Gone to a movie) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Played board or card games in person with others?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Played a team sport?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Hosted visitors in your home) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Left your home to run an errand?) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Driven in a car by yourself) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Driven in a car with someone else) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Taken a bus) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Taken a rideshare) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Taken a flight) | Str | Unchecked, Checked | Unchecked |
| During the first week of Spring break, what activities did you participate in: (choice=Made facecalls (facetime, whatsapp, zoom)  with Family or Friends) | Str | Unchecked, Checked | Unchecked |
| Did you travel out of Austin during the second week of spring break (March 22-29th)? | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (check all that apply) (choice=Vacation) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (check all that apply) (choice=Return home) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (check all that apply) (choice=Work) | Str | Unchecked, Checked | Unchecked |
| Reason(s) for travel (check all that apply) (choice=Other) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Cooked dinner at home?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Picked up take-out food) | Str | Unchecked, Checked | Unchecked |
|  During the second week of Spring break, what activities did you participate in: (choice=Had take-out food delivered to your home?)| Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Had groceries delivered to your home?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a grocery store?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a pharmacy?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a doctors office) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a hospital?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to work outside the home) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone for a walk outside) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone outside to workout) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a park?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a City trail to walk or run?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a lake, river or stream to swim?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a beach?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a gym?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Exercised at home?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Eaten in a restaurant?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone into a bar) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a gathering or meeting of 10 or more people) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Gone to a movie) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Played board or card games in person with others?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Played a team sport?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Hosted visitors in your home) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Left your home to run an errand?) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Driven in a car by yourself) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Driven in a car with someone else) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Taken a bus) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Taken a rideshare) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Taken a flight) | Str | Unchecked, Checked | Unchecked |
| During the second week of Spring break, what activities did you participate in: (choice=Made facecalls (facetime, whatsapp, zoom)  with Family or Friends) | Str | Unchecked, Checked | Unchecked |
| Did you change your activities during the first week of spring break in response to the COVID-19 Outbreak? | Str | Yes, No | Yes |
| How (check all that apply)? (choice=Cancelled vacation) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Shortened vacation) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Limited travel) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Changed activities) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Social distancing) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Other) | Str | Unchecked, Checked | Unchecked |
| Did you change your activities during the second week of spring break in response to the COVID-19 Outbreak? | Str | Yes, No | Yes |
| How (check all that apply)? (choice=Cancelled vacation) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Shortened vacation) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Limited travel) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Changed activities) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Social distancing) | Str | Unchecked, Checked | Unchecked |
| How (check all that apply)? (choice=Other) | Str | Unchecked, Checked | Unchecked |
| Complete? | Str | Incomplete, Complete | Incomplete |

## Fitbit Home Address 
Similar to the instruction "surveys", this "survey" was sent out twice in Event 2 and 3 (RedCAP). When downloading the data, the phone number and address fields remain hidden since they contain sensitive information. 

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 9 |
| Event Name | Str | NA | Event 1 |
| Survey Timestamp | Str | NA | 04/30/20 15:54 | 
| Please enter your phone number. | Str | NA | (999) 999-9999 |
| Please enter your current home address. | Str | NA | 1234 Wickersham Lane Paris, TX 77777 |
| Complete? | Str | Complete, Incomplete | Complete |


## Crisis Adult Selfreport Baseline Form
This form also contains questions that RedCAP does not export because they contain sensitive information - primarily deomgraphic information.

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 1 |
| Event Name | Str | NA | Event 3 |
| Timestamp | Str | NA | 6/22/19 13:19 |
| Country | Str | NA | United States |
| State/Providence/Region | Str | NA | TX |
| Your Age (years) | Int | NA | 27 |
| Please specify your sex | Str | NA | Male |
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=England, Ireland, Scotland or Wales) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Australia - not of Aboriginal or Torres Strait Islander descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Australia - of Aboriginal or Torres Strait Islander descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=New Zealand - not of Maori descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=New Zealand - of Maori descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Northern Europe including Sweden, Norway, Finland and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Western Europe including France, Germany, the Netherlands and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Southern Europe including Italy, Greece, Spain, Portugal and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Eastern Europe including Russia, Poland, Hungary and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Middle East including Lebanon, Turkey and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Eastern Asia including China, Japan, South Korea, North Korea, Taiwan and Hong Kong) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=South-East Asia including Thailand, Malaysia, Indonesia, Singapore and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=South Asia including India, Pakistan, Sri Lanka and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Polynesia, Micronesia or Melanesia including Tonga, Fiji, Papua New Guinea and surrounding countries) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Africa) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=North America - not of First Nations, Native American, Inuit or Métis descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=North America - of First Nations, Native American, Inuit or Métis descent) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Central or South America) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents) come from?  You may select as many choices as you need. (choice=Dont know) | Str | Unchecked, Checked | Unchecked | 
| Thinking about what you know of your family history, which of the following best describes the geographic regions where your ancestors (i.e. your great-great-grandparents)
| Are you of Hispanic or Latino descent - that is, Mexican, Mexican American, Chicano, Puerto Rican, Cuban, South or Central American or other Spanish culture or origin? | Str | Yes, No | No |
| Are you currently working or in school? (choice=Working for pay) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=On leave) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=Unemployed and looking for a job) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=Retired) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=Staying at home / homemaker) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=Disabled) | Str | Unchecked, Checked | Unchecked |
| Are you currently working or in school? (choice=Enrolled in school/college/university) | Str | Unchecked, Checked | Unchecked |
| Have you served in the military? | Str | Yes, No | No |
| Which best describes the area in which you live? | Str | Large city; Suburbs of a large city; Small city; Town or village | <- |
| What is the highest level of education YOU completed? | Str | High school diploma or GED; Some school beyond college; Some college or 2-year degree; 4-year college graduate; Graduate or professional degree | <- |
| What is the highest level of education your MOTHER completed? | Str | High school diploma or GED; Some school beyond college; Some college or 2-year degree; 4-year college graduate; Graduate or professional degree | <- |
| What is the highest level of education your FATHER completed? | Str | High school diploma or GED; Some school beyond college; Some college or 2-year degree; 4-year college graduate; Graduate or professional degree | <- |
| How many people currently live in your home (excluding yourself)? | Int | NA | 2 |
| Please specify your relationship to the people in your home (check all that apply): (choice=Partner/Spouse) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Parent(s)) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Grandparent(s)) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Siblings) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Children) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Other relatives) | Str | Unchecked, Checked | Unchecked |
| Please specify your relationship to the people in your home (check all that apply): (choice=Unrelated person) | Str | Unchecked, Checked | Unchecked |
| How many rooms (total) are in your home? | Int | NA | 3 |
| Are you covered by health insurance? | Str | Yes, individual; Yes, Medicaid or CHIP; Yes, employer-sponsored; Yes, other; No | <- |
| In the 3 months prior to the Coronavirus/COVID-19 crisis in your area, did you or your family receive money from government assistance programs like welfare, Aid to Families with Dependent Children, General Assistance, or Temporary Assistance for Needy Families? | Str | Yes, No | No |
| How would you rate your overall physical health? | Str | Excellent, Very Good, Good, Fair, Poor | Fair |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Seasonal allergies) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Asthma or other lung problems) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Heart problems) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Kidney problems) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Immune disorder) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Diabetes or high blood sugar) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Cancer) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Arthritis) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Frequent or very bad headaches) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Epilepsy or seizures) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Serious stomach or bowel problems) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Serious acne or skin problems) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Emotional or mental health problems such as Depression or Anxiety) | Str | Unchecked, Checked | Unchecked |
| Has a health professional ever told you that you had any of the following health conditions (check all that apply)?  (choice=Problems with alcohol or drugs) | Str | Unchecked, Checked | Unchecked |
| How tall are you? (inches) | Float | NA | 63.5 |
| How much do you weigh? (pounds) | Float | NA | 122 |
| How would you rate your overall Mental/Emotional health before the Coronavirus/COVID-19 crisis in your area? | Str | Excellent, Very Good, Good, Fair, Poor | Fair |
| ...have you been exposed to someone likely to have Coronavirus/COVID-19? (check all that apply) (choice=Yes, someone with positive test) | Str | Unchecked, Checked | Unchecked |
| ...have you been exposed to someone likely to have Coronavirus/COVID-19? (check all that apply) (choice=Yes, someone with medical diagnosis, but no test) | Str | Unchecked, Checked | Unchecked |
| ...have you been exposed to someone likely to have Coronavirus/COVID-19? (check all that apply) (choice=Yes, someone with possible symptoms, but no diagnosis by doctor) | Str | Unchecked, Checked | Unchecked |
| ...have you been exposed to someone likely to have Coronavirus/COVID-19? (check all that apply) (choice=No) | Str | Unchecked, Checked | Unchecked |
| ...have you been suspected of having Coronavirus/COVID-19 infection? | Str | No symptoms or signs; Yes, have had some possible symptoms, but no diagnosis by doctor | <- |
| ...have you had any of the following symptoms? (check all that apply) (choice=Fever) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Cough) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Shortness of breath) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Sore throat) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Fatigue) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Loss of taste or smell) | Str | Unchecked, Checked | Unchecked |
| ...have you had any of the following symptoms? (check all that apply) (choice=Other) | Str | Unchecked, Checked | Unchecked |
| ...has anyone in your family been diagnosed with Coronavirus/COVID-19? (check all that apply) (choice=Yes, member of household) | Str | Unchecked, Checked | Unchecked |
| ...has anyone in your family been diagnosed with Coronavirus/COVID-19? (check all that apply) (choice=Yes, non-household member) | Str | Unchecked, Checked | Unchecked |
| ...has anyone in your family been diagnosed with Coronavirus/COVID-19? (check all that apply) (choice=No) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Fallen ill physically) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Hospitalized) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Put into self-quarantine with symptoms) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Put into self-quarantine without symptoms (e.g., due to possible exposure)) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Lost job) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Reduced ability to earn money) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=Passed away) | Str | Unchecked, Checked | Unchecked |
| ...have any of the following happened to your family members because of Coronavirus/COVID-19? (check all that apply)  (choice=None of the above) | Str | Unchecked, Checked | Unchecked |
| ...being infected? | Str | Not at all, Slightly, Moderately, Very, Extremely | Slightly |
| ...friends or family being infected? | Str | Not at all, Slightly, Moderately, Very, Extremely | Slightly |
| ...your physical health being inﬂuenced by Coronavirus/COVID-19? | Str | Not at all, Slightly, Moderately, Very, Extremely | Slightly |
| ...your Mental/Emotional health being inﬂuenced by Coronavirus/COVID-19? | Str | Not at all, Slightly, Moderately, Very, Extremely | Slightly |
| How much are you reading, or talking about Coronavirus/COVID-19? | Str | Never, Rarely, Occasionally, Often, Most of the Time | Rarely |
| Has the Coronavirus/COVID-19 crisis in your area led to any positive changes in your life?   | Str | None, Only a few, Some | None |
| ...if you attend school, has your school building been closed? | Str | Yes, No | No |
| Are classes in session? | Str | Yes, No | No |
| Are you attending classes in-person?  | Str | Yes, No | No |
| Have classes resumed online? | Str | Yes, No | Yes |
| Do you have easy access to the internet and a computer? | Str | Yes, No | Yes |
| Are there assignments for you to complete?  | Str | Yes, No | Yes |
| Are you able to receive meals from the school? | Str | Yes, No | Yes |
| ...if you are working, has your workplace closed? | Str | Yes, No | Yes |
| ...if you are working, have you been able to telework or work from home?  | Str | Yes, No | Yes |
| ...how many people, from outside of your household, have you had an in-person conversation with? | Int | NA | 5 |
| ...how much time have you spent going outside of the home (e.g., going to stores, parks, etc.)? | Str | Not at all; 1-2 days per week; A few days per week; Several days perweek; Every day | Not at all |
| ...how stressful have the restrictions on leaving home been for you? | Str | Not at all, Slightly, Moderately, Very, Extremely | Slightly |
| ...have your contacts with people outside of your home changed relative to before the Coronavirus/COVID-19 crisis in your area? | Str | A lot less,A little less, About the same, A little more, A lot more | <- |
| ...how much difﬁculty have you had following the recommendations for keeping away from close contact with people? | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
|  | Str | | <- |
| ...cigarettes or other tobacco? | Str | | <- |
| ...marijuana/cannabis (e.g., joint, blunt, pipe, bong)? | Str | | <- |
| ...opiates, heroin, or narcotics? | Str | | <- |
| ...other drugs including cocaine, crack, amphetamine, methamphetamine, hallucinogens, or ecstasy? | Str | | <- |
| ...sleeping medications or sedatives/hypnotics? | Str | | <- |
| ...how many hours per night did you sleep on average?   | Str | | <- |
| ...how many days per week did you exercise (e.g., increased heart rate, breathing) for at least 30 minutes?       | Str | | <- |
| ...how many days per week did you spend time outdoors?       | Str | | <- |
| ...how worried were you generally?       | Str | | <- |
| ...how happy versus sad were you?       | Str | | <- |
| ...how much were you able to enjoy your usual activities?  | Str | | <- |
| ...how relaxed versus anxious were you? | Str | | <- |
| ...how fidgety or restless were you? | Str | | <- |
| ...how fatigued or tired were you? | Str | | <- |
| ...how well were you able to concentrate or focus? | Str | | <- |
| ...how irritable or easily angered have you been? | Str | | <- |
| ...how lonely were you? | Str | | <- |
| ...to what extent did you have negative thoughts, thoughts about unpleasant experiences or things that make you feel bad? | Str | | <- |
| ...watching TV or digital media (e.g., Netflix, YouTube, web surfing)?    | Str | | <- |
| ...using social media (e.g., Facetime, Facebook, Instagram, Snapchat, Twitter, TikTok)?   | Str | | <- |
| ...playing video games?   | Str | | <- |
| ...alcohol? | Str | | <- |
| ...vaping? | Str | | <- |
| ...cigarettes or other tobacco? | Str | | <- |
| ...marijuana/cannabis (e.g., joint, blunt, pipe, bong)? | Str | | <- |
| ...opiates, heroin, or narcotics? | Str | | <- |
| ...other drugs including cocaine, crack, amphetamine, methamphetamine, hallucinogens, or ecstasy? | Str | | <- |
| ...sleeping medications or sedatives/hypnotics? | Str | | <- |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Resource room) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Tutoring) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Mentoring programs) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=After school activity programs) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Volunteer programs) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Psychotherapy) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Psychiatric care) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Occupational therapy) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Physical therapy) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Speech/language therapy) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Sporting activities) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Medical care for chronic illnesses) | Str | Unchecked, Checked | Unchecked |
| Which of the following supports were in place for you before the Coronavirus/COVID-19 crisis in your area and have been disrupted over the PAST TWO WEEKS? (check all that apply) (choice=Other) | Str | Unchecked, Checked | Unchecked |
| Complete? | Str | Incomplete, Complete | Complete |

## Weekly Behaviors

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 1 |
| Event Name | Str | NA | Event 3 |
| Survey Timestamp | Str | NA | 12/5/19 16:18 |
| 1.	In the past week, on average, how many hours PER DAY did you spend in your bedroom? (# of hours) | Float | NA | 1.5 |
| 2. ...in the kitchen? (# of hours PER DAY in past week) | Float | NA | 0.5 |
| 3. ...in the living room? (# of hours PER DAY in past week) | Float | NA | 12 |
| 4. ...in your backyard or back patio/balcony? (# of hours PER DAY in past week) | Float | NA | 5 |
| 5. ...In your front yard or front patio/balcony? (# of hours PER DAY in past week) | Float | NA | 0 |
| Has your diet changed or been affected by COVID-19 social distancing? EATING: | Str | less than usual, about the same, more than usual | <- |
| Has your diet changed or been affected by COVID-19 social distancing? DRINKING WATER: | Str | less than usual, about the same, more than usual | <- |
| Has your diet changed or been affected by COVID-19 social distancing? DRINKING ALCOHOL:  | Str | less than usual, about the same, more than usual | <- |
| Has your diet changed or been affected by COVID-19 social distancing? DIET: | Str | less than usual, about the same, more than usual | <- |
| Complete? | Str | Complete, Incomplete | <- |

## Covid Survey

| Field Name | Variable Type | Answer Choices | Example |
| --- | --- | --- | --- |
| Record ID | Int | NA | 1 |
| Event Name | Str | NA | Event 3 |
| Survey Timestamp | Str | NA | 12/5/19 16:27 |
| Since the University's closure, are you living in the U.S.? | Str | Yes, No | <- |
| What is your employment status? | Str | unemployed, looking for work; student, not working; student, part time work; student, full time work | <- |
| What is your marital status? | Str | Never Married, Currently cohabiting or Married | <- |
| Compared to others your age, how would you rate your health? | Str | Poor, Below Average, Average, Good, Excellent | <- |
| Do you have health insurance? | Str | Yes (including parent or government provided insurance programs); No | <- |
| How would you characterize your political orientation? | Str | Extremely Liberal, Somewhat Liberal, In the middle, Somewhat Conservative, Extremely Conservative | <- |
| Have you been tested for the COVID-19 virus? | Str | No, I have not been tested; Yes, I tested negative; Yes, I tested positive | <- |
| Have you experienced symptoms that you feel may be related to COVID-19? | Str | No; Yes, but I am pretty certain it is not COVID-19; Yes, I feel it could be COVID-19 | <- |
| Tested positive for the COVID-19 virus? | Str | Yes, No | <- |
| Has died from COVID-19? | Str | Yes, No | <- |
| Self isolated (staying at home, avoiding public places) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- |
| Been using masks or gloves while going out | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- |
| Practiced other preventative measures (e.g., washing hands) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- |
| Getting COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Dying from COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Family members getting COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Unknowingly infecting others | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Having COVID-19 (even though you're pretty sure you don't) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Interesting | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Annoying | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Anxiety-provoking | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Terrifying | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Liberating | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Exciting | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Depressing | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Boring | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Lost or about to lose my job due to COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Have job but will lose significant income due to COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Working remotely due to COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Currently working in job that involves working face to face with the general public | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Paying rent/utility bills | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Affording food | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Affording medical bills | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Childcare | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Losing your job in the next few months | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Family members | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Kids | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| People related to your work (employees, students, clients, customers, etc.) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Travel plans | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Going to restaurants/bars/coffee shops and other public places | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Education or the education of family members living with you | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Daily routine | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Work patterns | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Social life | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Living arrangements | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Your family | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| Your friends | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| The people in your workplace or school (including co-workers, customers, clients, and students) | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| Your neighbors | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| Your city | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| Your country | Str | Much less connected; A little less connected; About the Same; A Little more connected; Much more connected | <- |
| Bring people together | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Tear us apart | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| People in the medical profession | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Asians | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Homeless people | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Service workers (e.g., waiters, bus drivers, cleaning staff...) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Europeans | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| People from Mexico and Central America | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Friends and neighbors | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Facebook | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Youtube | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Twitter | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Other social media (e.g., Reddit, Instagram) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Government sources (CDC, NHS, etc.) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Print or online news | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Radio, TV, and podcasts | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| People are making too big a deal about COVID-19 | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| People aren't taking COVID-19 seriously enough | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| COVID-19 might be the result of a foreign government conspiracy | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| COVID-19 is a result of climate change | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| COVID-19 is just a normal thing that happens every now and then | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| COVID-19 is a result of immigrants coming to your country | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| In the last week, to what degree have you been involved in panic buying (including hoarding and buying excessive amounts) | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| Went to a crowded place for recreational purposes (e.g., restaurant, bar, coffee shop, party) | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Went to a crowded place for academic or work purposes (e.g., conference, lecture, meeting) | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Used public transit (e.g., bus, airplane, train) | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Went to a supermarket, pharmacy or other store | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Family members | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Friends | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| The people in your workplace or school (including co-workers, customers, clients, and students) | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Others (including service workers) | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Family members | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Friends | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| The people in your workplace or school (including co-workers, customers, clients, and students) | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Others (including service workers) | Mixed | 0; 1-2 persons; 3-4 persons; 5-9 persons; Over 10 persons | <- |
| Awake, in your house/apartment/dorm | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Awake, in the home of friend(s) or family | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| On social media | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Outdoors | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Sleeping | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Exercising | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Playing video/computer/online games | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Eating | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Under the influence of alcohol or other recreational substances | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Watching in-home movies, videos, or TV | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Reading books/magazines/stories (not related to COVID-19) | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Shopping online | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Feeling depressed | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Feeling alone | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Working (including working remotely) | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Cooking and other housework | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| House or yard maintenance | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Learning a new skill | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Reading/learning about COVID-19 | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Watching/listening to news about COVID-19 | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Talking/texting/communicating with others about COVID-19 | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Helping others to cope with COVID-19 | Mixed | 0; Up to 1 hour; 1-3 hours; 3-7 hours; 8 hours or more | <- |
| Argue or have a conflict with someone | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Get angry with a friend or family member | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Reach out to check on someone | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Offer help to someone | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Smoke | Str | Not once; Once; 2-3 times; 3-10 times | <- |
| Overall, to what degree are you happy that the COVID-19 outbreak occurred in your lifetime? | Str | A great deal; A lot; A moderate amount; A little; Not at all | <- | 
| How long do you think the COVID-19 outbreak would last? | Str | One month; Two months; Three months; Four months; Five months; Six months to one year; One to two years; More than two years | <- |
| Do you with to share your Reddit or Twitter handles? | Str | No, Yes | No |
| Complete? | Str | Incomplete, Complete | Complete |
