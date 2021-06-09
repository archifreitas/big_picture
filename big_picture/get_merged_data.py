import pandas as pd
import glob

# merge .csv data from a folder
# Arguments to change based on data location and filename
# Data for loading .csv

rel_path_input = "../raw_data/data_12k"

def get_data(rel_path_input):

    files = glob.glob(rel_path_input + "*.csv")
    df = pd.DataFrame()

    for file in files:
        csv = pd.read_csv(file)
        df = df.append(csv)

    return df

# example
# get_merged_data("../raw_data/data_12k/", "../raw_data/", "test.csv")