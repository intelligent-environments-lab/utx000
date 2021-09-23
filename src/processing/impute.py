
import pandas as pd

# Iterative Imputer for MICE
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

class TestData:

    def __init__(self,pt,data_dir,params=["co2","pm2p5_mass","tvoc","temperature_c","rh"]):
        self.pt = pt
        self.data_dir = data_dir
        self.params = params

        self.data = self.import_example_data(self.pt,self.data_dir,self.params)

    def import_example_data(self,pt,data_dir="../",params=["co2","pm2p5_mass","tvoc","temperature_c","rh"]):
        """
        Imports example data
        
        Parameters
        ----------
        pt : str
            id of the participant to import for
        data_dir : str
            path to the "data" directory in the utx000 project
            
        Returns
        -------
        data : DataFrame
            exemplary ieq data from participant
        """
        
        try:
            data = pd.read_csv(f"{data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",
                            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        except FileNotFoundError:
            print("No filename for participant", pt)
            return pd.DataFrame()
        
        return data[params]

class Impute:

    def __init__(self,pt,data_dir,consecutive=False):
        # Class Vars
        # ----------
        self.pt = pt
        self.data_dir = data_dir
        # Loading Data
        # ------------
        # Missing Data
        percent = input("Percent: ") # common to both
        if consecutive:
            param = input("Parameter: ")
            period = input("Period (in minutes): ")
            self.load_data_consecutive_random(param,percent,period)
        else:
            self.load_data_random(percent)
        # Base Data
        self.base = pd.read_csv(f"{self.data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",
                                index_col="timestamp",infer_datetime_format=True)
        self.base = self.base[[col for col in self.missing.columns]]

    def load_data_random(self,percent):
        """
        Loads in the randomly removed data
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_all-p{percent}-{self.pt}.csv",parse_dates=["timestamp"],
                                    index_col="timestamp",infer_datetime_format=True)
        except FileNotFoundError as e:
            print(e)
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tpercent:\t{percent}")

    def load_data_consecutive_random(self,param,percent,period):
        """
        Loads in the randomly removed, consecutive data for param
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_periods_all-{param}-p{percent}-{period}mins-{self.pt}.csv",
                                    parse_dates=["timestamp"],index_col="timestamp",infer_datetime_format=True)
        except FileNotFoundError:
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tparam:\t{param}\n\tpercent:\t{percent}\n\tperiod:\t{period}")

    def mice(self,estimator=None):
        """
        Imputes missing data with Mutiple Iterations of Chained Equations (MICE)
        """
            
        imp = IterativeImputer(estimator=estimator,max_iter=30,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.mice_imputed = pd.DataFrame(imp.transform(self.missing),index=self.base.index,columns=self.base.columns)

    def miss_forest(self):
        """
        Imputes missing data with missForest
        """
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10),max_iter=10,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.rf_imputed = pd.DataFrame(imp.transform(self.missing),index=self.base.index,columns=self.base.columns)

    def gans(self):
        """
        Imputes missing data with Generative Adversarial Networks (GANs)
        """
        pass