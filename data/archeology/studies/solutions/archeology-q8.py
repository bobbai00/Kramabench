#!/usr/bin/env python
# coding: utf-8

import pandas as pd

data_path = "../input"



# question: how many sources were used in the dataset
roman_path = "{}/roman_cities.csv".format(data_path)

roman_df = pd.read_csv(roman_path)



# Get the bibliography for each city, and do data cleaning to remove any bad values
filtered_df = roman_df[roman_df["Select Bibliography"].notna()]
roman_sources = filtered_df["Select Bibliography"]



# Separate the sources by the semicolons and find all of the unique ones

sources = set()
index = 0
for values in roman_sources:
    values = values.replace(".", "").split(";")
    for value in values:
        sources.add(value)

print(len(sources))




