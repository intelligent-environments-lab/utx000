UTx000
==============================

UTX000 is a study that gathers from multiple individuals/studies and contains a wealth of moment-by-moment data on a person and their envrionment. The project is an arm of [**Whole Communities, Whole Health Iniative - a UT Grand Challenge**](https://bridgingbarriers.utexas.edu/whole-communities-whole-health/). The goals of this particular project are to:

* reconcile multiple data modalities
* establish connections between variables from the same and disparate data modalities
* provide actionable results for participants

Interested in the nitty-gritty of the project? Check out more on the [Wiki page](https://github.com/intelligent-environments-lab/utx000/wiki).

Data Modalities
------------
The project uses data gathered from four main sources:

* UT's IEL Indoor Air Quality (IAQ) Beacon
* Onnela Labs's [Beiwe](http://wiki.beiwe.org/wiki/Main_Page) Platform
* Fitbit Wearable Devices gathered on the [Fitabase](https://www.fitabase.com) Platform
* Surveys administered by UT's REDCap Platform hosted the by Population Research Center

Other sources of data include:
* Hair and saliva samples
* Home dust samples

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data (not included)
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    |   ├── purgatory      <- Raw data with inconsistent formatting.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── images             <- Study-related images
    │
    ├── notebooks          <- Jupyter notebooks for the majority of analysis - see readme in the folder for more details
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials - see Wiki too
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features 
    │   │   └── build_features.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── videos             <- Study-related videos


Contribute
----------

- Issue Tracker: [Issues](github.com/intelligent-environments-lab/utx000/issues)
- Source Code: [SRC](github.com/intelligent-environments-lab/utx000/tree/master/src)

Support
-------

If you are having issues, please contact the project author Hagen Fritz <br>
Email: HagenFritz@utexas.edu

License
-------

The project is licensed under the MIT license.

-------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
