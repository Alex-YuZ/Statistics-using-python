import pandas as pd
import numpy as np
from t_test import between_t_test
import argparse


def load_data(filename):
    df = pd.read_csv(filename)
    
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
    parser = argparse.ArgumentParser(description="Choose a dataset to run between-group t-Test")
    
    parser.add_argument('--dataset', type=str, help='the dataset you want to execute t-test on')
    
    args = parser.parse_args()
    main(args.dataset)