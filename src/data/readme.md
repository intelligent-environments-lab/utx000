# Data File Structure

The data file structure for this project is necessarily complicated due to:

- the evolving nature of how data are collected
- how data are downloaded from their various sources

The first point is primarily aimed at the beacon. Initially referred to as the BEVO Beacon, this model has been decomissioned in favor of the newer IAQ Beacon. For the foreseeable future, the plan remains to use the IAQ Beacon and thus the file structure will remain intact. However, since data are included from the UT1000 and UT2000 studies, we must keep a more specified path structure. 

The second point is aimed at data files downloaded from Fitabase, Beiwe, and REDCap.

- Downloads from Fitabase are agnostic to the study they were apart of and rather than renaming the file to reflect the study, we simply provide the filepath to illustrate this.
- Beiwe data are organized according to the ID of the participant. While these IDs are known and kept in a separate document, subdividing the data through a file structure is easier than having to cross-reference a master document.
- Information from REDCap is typically in poor fashion and most data files are "hand"-crafted from the information. It is just as easy to rename the file or drop it in the correct location. 

## Data File Structure

    ├── raw                     <- A default Sphinx project; see sphinx-doc.org for details
        ├── study1
        ├── study2              <- multiple studies exist in the directory - only an example
            ├── beiwe           <- beiwe data
                ├── pid1
                ├── pid2
                    ├── survey_answers
                        ├── morning_survey_id
                        ├── evening_survey_id
                        ├── weekly_survey_id
            ├── beacon
            ├── fitbit
            ├── beiwe
        ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
