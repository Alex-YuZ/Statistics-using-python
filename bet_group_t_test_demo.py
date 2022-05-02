import pandas as pd
import numpy as np
from t_test import between_t_test


# between-group acne medicine demo
acne = pd.read_csv('acne.csv')
print(acne)

g1 = acne.group1[acne.group1.notna()].to_numpy()
g2 = acne.group2[acne.group2.notna()].to_numpy()

# between-group avg_food_price demo
food = pd.read_csv('avg_food_price.csv')
print(food)

g = food.gettysburg[food.gettysburg.notna()].to_numpy()
w = food.wilma[food.wilma.notna()].to_numpy()


if __name__ == '__main__':
    between_t_test(g, w)