import pandas as pd

import os

class MIC():

    def __init__(self,data,**kwargs):
        self.data=data
        if "path_to_data" in kwargs.keys():
            self.path_to_data = kwargs["path_to_data"]
        else:
            self.path_to_data = "./data_files"
    
    # setters
    def set_data(self,data):
        """sets the class's data with the given dataframe"""
        self.data = data

    def create_mic_file(self,pt=1,by_id="beacon",scale=False,cols_to_drop=[]):
        """
        Creates a file for the given participant in the MIC-specific file format
        
        Inputs
        ------
        pt: integer or string specifying the participant identifier
        by_id: string of [beiwe,beacon,redcap,fitbit] that specifies which utx000 ID type to consider
        scale: boolean to scale each of the variables between 0 and 1
        cols_to_drop: list of additional columns to drop
        
        Generates and saves the file
        """
        data_by_pt = self.data[self.data[by_id] == pt]
        data_by_pt.dropna(inplace=True)
        unecessary_columns = [col for col in ["timestamp","beacon","beiwe","fitbit","redcap"] if col != by_id]
        for label in unecessary_columns + cols_to_drop:
            try:
                data_by_pt.drop(label,axis=1,inplace=True)
            except KeyError:
                pass
            
        data_by_pt.reset_index(drop=True,inplace=True)
        data_by_pt.T.to_csv(fr"{self.path_to_data}/temp.txt",header=True,index=True,sep='\t')

    def calculate_strength(self,**kwargs):
        """"""
        if "data_file" in kwargs.keys():
            fname = kwargs["data_file"]
        else:
            fname = "temp.txt"

        os.system(f"mictools null {self.path_to_data}/{fname} {self.path_to_data}/null_dist.txt")
        os.system(f"mictools pval {self.path_to_data}/{fname} {self.path_to_data}/null_dist.txt {self.path_to_data}")
        os.system(f"mictools adjust {self.path_to_data}/pval.txt {self.path_to_data}")
        os.system(f"mictools strength {self.path_to_data}/{fname} {self.path_to_data}/pval_adj.txt {self.path_to_data}/strength.txt")

    def run(self):
        pass

def main():
    pass