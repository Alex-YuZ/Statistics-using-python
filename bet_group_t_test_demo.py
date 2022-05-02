import pandas as pd
import numpy as np
from t_test import between_t_test

def load_data(filename):
    df = pd.read_csv(filename)
    s1 = df.s1[df.s1.notna()].to_numpy()
    s2 = df.s2[df.s2.notna()].to_numpy()
    
    return df, s1, s2

# between-group acne medicine demo
acne, g1, g2 = load_data('acne.csv')
print(acne)


# between-group avg_food_price demo
food, g, w = load_data('avg_food_price.csv')
print(food)


if __name__ == '__main__':
    between_t_test(g, w)