import pandas as pd
import glob

# merge .csv data from a folder
# Arguments to change based on data location and filename
# Data for loading .csv

def get_data(rel_path_input):

    files = glob.glob(rel_path_input + "*.csv")
    df = pd.DataFrame()

    for file in files:
        csv = pd.read_csv(file)
        df = df.append(csv)
    
    return df

# example
# rel_path_input = "../raw_data/data_12k/"
# get_merged_data("../raw_data/data_12k/", "../raw_data/", "test.csv")

if __name__ == '__main__':
    print("Testing ...")
    df = get_data("../raw_data/data_12k/")
    
    print(f"Final Shape: {df.shape}")
    print("Done!")