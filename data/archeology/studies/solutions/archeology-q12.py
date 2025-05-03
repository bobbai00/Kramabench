#!/usr/bin/env python
# coding: utf-8


import pandas as pd

data_path = "../input"



city_path = "{}/worldcities.csv".format(data_path)
city_df = pd.read_csv(city_path)
countries = city_df["country"].unique().tolist()
countries = set(countries)



# Load and process the ucdp dataset

path = "{}/conflict_brecke.csv".format(data_path)
data = pd.read_csv(path)
processed_rows = []
for index, row in data.iterrows():
    name = row["Conflict"]
    if "-" in name:
        name = name.split("-")
        a = name[0]
        b = name[1]
    elif "and" in name:
        name = name.split("and")
        a = name[0]
        b = name[1]
    else: # assume self-conflict
        a = name
        b = name

    start_year = int(row["StartYear"])
    end_year = int(row["EndYear"])

    def get_matching_word(sentence):
        matches = [word for word in countries if word.lower() in sentence.lower()]
        return matches[0] if matches else None
                   
    a = get_matching_word(a)
    b = get_matching_word(b)

    if a == None or b == None:
        continue

    a, b = sorted([a, b])

    processed_rows.append({
        "a": a,
        "b": b,
        "start": start_year,
        "end": end_year,
    })

filtered_data = pd.DataFrame(processed_rows)



# Combine the two and output the total number
sorted_data = filtered_data.sort_values(by=["a", "b", "start", "end"], ascending=[True, True, True, False]).reset_index(drop=True)
def check_for_overlaps(df):
    valid_rows = []
    valid_rows.append({"a": df.iloc[0]["a"], "b": df.iloc[0]["b"], "start": df.iloc[0]["start"], "end": df.iloc[0]["end"]})
    for i in range(1, len(df)):
        a = df.loc[i]["a"]
        b = df.loc[i]["b"]
        start = df.loc[i]["start"]
        end = df.loc[i]["end"]
        if a == valid_rows[-1]["a"] and b == valid_rows[-1]["b"]:
            if valid_rows[-1]["start"] <= start and valid_rows[-1]["end"] >= start:
                start = valid_rows[-1]["end"] + 1
                if start > end:
                    continue

        valid_rows.append({"a": a, "b": b, "start": start, "end": end})
    
    return pd.DataFrame(valid_rows)

# Apply the overlap check
check = check_for_overlaps(sorted_data)
print(len(check))




