#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
data_path = "../input"



# What is the average latitude of capital cities?

city_path = "{}/worldcities.csv".format(data_path)

df = pd.read_csv(city_path)



capitals = df[df["capital"] == "primary"].reset_index(drop=True)
print(len(capitals))


countries = set(capitals["country"])
print(len(countries))



values = []
for country in countries:
    country_capitals = capitals[capitals["country"] == country].reset_index(drop=True)
    if len(country_capitals) > 1:
        #max_population_idx = country_capitals["population"].idxmax()
        population_values = country_capitals["population"].values
        max_population_idx = np.argmax(population_values)
        values.append(country_capitals["lat"].values[max_population_idx])
    else:
        values.append(country_capitals["lat"].values[0])

print("Number of capital cities: ", len(values))


print(np.mean(values))



