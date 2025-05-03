#!/usr/bin/env python
# coding: utf-8


import pandas as pd

data_path = "../input"



# Which country has the highest average population in its cities?

city_path = "{}/worldcities.csv".format(data_path)

df = pd.read_csv(city_path)


# Compute the average, and find the highest one

countries = df.groupby("country")["population"].mean()
index = countries.idxmax()
print(index)



