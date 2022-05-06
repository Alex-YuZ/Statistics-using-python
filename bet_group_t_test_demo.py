import pandas as pd
import numpy as np
from t_test import between_t_test
import argparse
from argparse import RawTextHelpFormatter


def load_data(filename):
    df = pd.read_csv("./HypothesisTest_data/"+filename)
    
    # Select non-null values
    mask0 = df.iloc[:, 0].notna()
    mask1 = df.iloc[:, 1].notna()
    
    # Convert to numpy arrays
    s0, s1 = df.iloc[:, 0][mask0].to_numpy(), df.iloc[:, 1][mask1].to_numpy()
    
    return df, s0, s1


def main(filename):
    df, s0, s1 = load_data(filename)
    print(df)
    between_t_test(s0, s1)
    

if __name__ == '__main__':
    descr_str = """
    Choose a dataset to run between-group t-Test.
      Available datasets: 
        (1)acne.csv
        (2)avg_food_price.csv
      
    Usage Example:
      If we want to take a look at how the between-group t-test work on dataset `acne.csv`,
      we can use the following codes in the terminal:
      
      ~$ python bet_group_t_test_demo.py --dataset acne.csv
      
    """
    parser = argparse.ArgumentParser(description=descr_str, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('--dataset', type=str, help='the dataset you want to execute t-test on')
    
    args = parser.parse_args()
    main(args.dataset)