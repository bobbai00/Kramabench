# Why Dataflow Succeeds When Code Agent Fails: Analysis of 10 Cases

## Overview

This analysis examines all 10 tasks where DataflowSystemO4Mini succeeded but CodeAgentSystemO4Mini failed. We identify **6 distinct failure patterns**, with **"Monolithic Multi-Step Blocks"** being the most common.

---

## NEW: Pattern 0 - Monolithic Multi-Step Blocks (Primary Pattern)

**This pattern appears in 6 out of 10 cases (60%)** and is the root cause enabling other failures.

### What It Means
The code agent combines multiple analytical steps into a single code block without verifying intermediate results. When an error occurs in the middle of the block, it goes undetected because:
1. No intermediate output is printed to verify correctness
2. The final output "looks reasonable" even if wrong
3. Bugs in early steps propagate silently to final answer

### Why Dataflow Avoids This
Dataflow's operator-based design forces modular execution:
- Each operator produces visible output
- Intermediate results can be inspected before proceeding
- Errors surface at specific operators, not buried in long code blocks

### Cases Exhibiting This Pattern

| Case | Multi-Step Block | Hidden Bug Location |
|------|------------------|---------------------|
| archeology-easy-11 | Load → Filter → GroupBy → Mean | Filter step: `notna()` instead of `== 'primary'` |
| environment-easy-4 | Filter → Count → Compare → Calculate | Compare step: `== 'NO'` instead of lowercase `'no'` |
| environment-easy-6 | Load 21 files → Concat → Filter → Count → Calculate | Compare step: `== "Yes"` case mismatch |
| biomedical-hard-5 | Load → Filter → Transform → Median | Transform step: `2 ** log2_value` wrong approach |
| legal-hard-23 | Load → Clean → Merge → Calculate → Max | Merge step: inner join dropped DC |
| wildfire-hard-21 | Load → Filter → Mean → Diff → Sort | Data source selection: used ZHVI instead of NOAA |

### Example: environment-easy-4

**Code Agent's Monolithic Block:**
```python
# All in one block - no intermediate verification!
wollaston_df = df[df['Beach Name'].isin(wollaston_names)]
total_samples = len(wollaston_df)
met_standards = (wollaston_df['Violation'] == 'NO').sum()  # BUG: should be 'no'
percentage_met = int((met_standards / total_samples) * 100)
print(percentage_met)  # Only output - by then it's too late
```

**Dataflow's Modular Approach:**
```python
# Operator 1: Filter beaches
def process(df): return df[df['Beach Name'].str.contains('Wollaston')]
# Execute and INSPECT output: see 'no'/'yes' values

# Operator 2: Count (after seeing actual data)
def process(filtered):
    return (filtered['Violation'].str.lower() == 'no').sum()  # Correct!
```

---

## NEW: Pattern 1 - Failed Error Recovery Due to State Loss (2 cases, 20%)

**The code agent tries to fix errors but keeps failing because each retry starts fresh without preserving progress from previous attempts.**

### What It Means
When the code agent encounters an error:
1. It attempts to fix the issue in the next step
2. But it must **re-execute all previous successful steps** from scratch
3. Variables and intermediate results from earlier steps are **lost**
4. After multiple failed retries, the agent often **gives up and guesses**

### Why Dataflow Avoids This
Dataflow's operator model preserves state:
- Each operator's output is **saved and persisted**
- When one operator fails, only that operator needs to be fixed
- Previous operators' results remain available
- The agent can resume from the last successful point

### Cases Exhibiting This Pattern

| Case | Retry Attempts | Errors Encountered | Final Outcome |
|------|----------------|-------------------|---------------|
| astronomy-hard-9 | 8 steps | ZeroDivisionError → FileNotFoundError → ValueError → ParseError | Gave up, guessed "15" (wrong, should be 24) |
| biomedical-hard-4 | 8 steps | header=8 wrong → header=9 wrong → README extract failed → column detection failed | Returned `[]` (wrong, should be ["FIGO Grade 2"]) |

### Example: astronomy-hard-9 (Error Cascade)

**Step-by-step failure cascade:**

```
Step 1: Parse TLE file → ZeroDivisionError (dt=0)
        ❌ Code fails, no state saved

Step 2: Try to read OMNI file → FileNotFoundError (wrong path)
        ❌ Must re-read TLE file, previous work lost

Step 4: List directory to find correct path → Success
        ✓ But TLE parsing still not done

Step 5: Parse both files together → ValueError (wrong column positions)
        ❌ Must start over again

Step 6-7: Try to find column spec in docs → Can't parse format
        ❌ Still no progress on actual computation

Step 8-9: GIVE UP → Guess "15"
        ❌ Wrong answer (correct: 24)
```

**What Dataflow Did Differently:**

```
Operator 1: Parse TLE file → Error
            Fix: Add dt>0 check → Re-run only this operator
            ✓ Output saved

Operator 2: Parse OMNI file → Error
            Fix: Correct file path → Re-run only this operator
            ✓ TLE output still available from Operator 1

Operator 3: Compute correlation
            ✓ Uses outputs from Operators 1 & 2
            → Correct answer: 24
```

### Example: biomedical-hard-4 (Repeated Parsing Failures)

**Step-by-step failure cascade:**

```
Step 1-2: Discover sheets in Excel file → Success
          ✓ Found: ['README', 'A-Variants', 'B-Novel Splice...']

Step 3: Try to read with header=8 → Column names not found
        ❌ Error, must re-read Excel file

Step 4: Try to read with header=9 → Column names not found
        ❌ Error, must re-read Excel file again

Step 5-6: Inspect actual column values → See data, not headers
          The header row varies per sheet!

Step 7: Try to extract headers from README → Got NaN
        ❌ README sheet has different structure

Step 8: Create dynamic header detection → Returns empty result
        ❌ Function worked but found no matching peptide
        → Returns [] (wrong, should be ["FIGO Grade 2"])
```

**The Core Problem:**
Each retry **re-reads the Excel file from scratch** instead of building on previous discoveries. The agent never realized that:
- Each sheet has headers at different row positions
- The peptide column name might have whitespace issues

### Why This Pattern Matters

| Code Agent | Dataflow |
|------------|----------|
| Each step starts fresh | State persisted between steps |
| Error = restart from scratch | Error = fix one operator |
| 8 retries = 8x file re-reads | Fix once, reuse cached results |
| Context lost between attempts | Full context preserved |
| Often gives up after multiple failures | Can iterate until correct |

---

## Updated Failure Pattern Categories

| Pattern | Count | Percentage | Description |
|---------|-------|------------|-------------|
| **Monolithic Multi-Step Blocks** | 6 | 60% | Multiple steps without intermediate verification |
| **Failed Error Recovery (State Loss)** | 2 | 20% | Retries fail because previous progress is lost |
| Semantic Misinterpretation | 3 | 30% | Misunderstood term meanings |
| Case Sensitivity Bugs | 3 | 30% | String comparison case mismatch |
| Calculation/Transformation Errors | 2 | 20% | Math or data transformation mistakes |
| Complex File Parsing Failures | 2 | 20% | Couldn't parse complex file formats |
| Incomplete Analysis | 2 | 20% | Gave up or skipped steps |
| Wrong Data Source | 1 | 10% | Used incorrect dataset |

*Note: Categories overlap - a case can have multiple patterns*

---

## Detailed Case Analysis

### Case 1: archeology-easy-11
**Patterns: Monolithic Block + Semantic Misinterpretation**

| Aspect | Value |
|--------|-------|
| Question | Average latitude of capital cities (largest pop per country) |
| Correct Answer | 17.4274 |
| Code Agent Answer | 17.1667 |

**The Monolithic Block (Step 2):**
```python
# 4 steps, no intermediate verification
cap_df = df[df['capital'].notna()]  # WRONG: includes admin, minor
cap_df_sorted = cap_df.sort_values('population', ascending=False, na_position='last')
selected = cap_df_sorted.drop_duplicates(subset='country', keep='first')
avg_lat = selected['lat'].mean()
print(round(avg_lat, 4))  # Only output - never verified filter result
```

**What Should Have Been Checked:**
- After filtering: How many rows? What capital types included?
- A simple `print(cap_df['capital'].value_counts())` would have revealed the issue

---

### Case 2: environment-easy-4
**Patterns: Monolithic Block + Case Sensitivity**

| Aspect | Value |
|--------|-------|
| Question | Percentage of Wollaston Beach samples meeting standards (2019-2023) |
| Correct Answer | 97 |
| Code Agent Answer | 33 |

**The Monolithic Block (Step 3):**
```python
wollaston_df = df[df['Beach Name'].isin(wollaston_names)]
total_samples = len(wollaston_df)
met_standards = (wollaston_df['Violation'] == 'NO').sum()  # BUG: data has 'no'
percentage_met = int((met_standards / total_samples) * 100)
```

**What Should Have Been Checked:**
- `print(wollaston_df['Violation'].unique())` → would show `['no', 'yes']`

---

### Case 3: environment-easy-6
**Patterns: Monolithic Block + Case Sensitivity**

| Aspect | Value |
|--------|-------|
| Question | Average exceedance rate for marine beaches (2002-2023) |
| Correct Answer | 5.14 |
| Code Agent Answer | 0.56 |

**The Monolithic Block (Step 2):**
```python
# 21 files loaded and processed in one block
dfs = [pd.read_csv(f) for f in files]
all_data = pd.concat(dfs, ignore_index=True)
marine = all_data[(all_data["Year"] >= 2002) & (all_data["Beach Type Description"] == "Marine")]
violation_samples = (marine["Violation"] == "Yes").sum()  # BUG: mixed case in data
exceedance_rate = violation_samples / total_samples * 100
```

**What Should Have Been Checked:**
- `print(marine['Violation'].value_counts())` → would reveal actual values

---

### Case 4: biomedical-hard-5
**Patterns: Monolithic Block + Calculation Error**

| Aspect | Value |
|--------|-------|
| Question | Median variants per Mbp for serous tumors |
| Correct Answer | 2.6563 |
| Code Agent Answer | 2.4241 |

**The Monolithic Block (Step 2):**
```python
serous_df = df[df["Histologic_type"].str.lower().str.contains("serous", na=False)]
serous_df["Variants_per_Mbp"] = 2 ** serous_df["Log2_variant_per_Mbp"]  # Transformation
median_variants = serous_df["Variants_per_Mbp"].median()
```

**What Should Have Been Checked:**
- Is the column really log2-transformed? Print sample values
- The correct answer (2.6563) suggests a different calculation approach

---

### Case 5: legal-hard-23
**Patterns: Monolithic Block + Data Loss in Merge**

| Aspect | Value |
|--------|-------|
| Question | State with highest report density (all report types) |
| Correct Answer | District of Columbia |
| Code Agent Answer | Arkansas |

**The Monolithic Block (Step 4):**
```python
df_id2 = df_id2.dropna(subset=['State'])  # May drop rows
df_fraud2 = df_fraud2.dropna(subset=['State'])  # May drop rows
df = pd.merge(df_id2[['State','Density_ID']],
              df_fraud2[['State','Density_FR']],
              on='State', how='inner')  # INNER JOIN - drops non-matching!
df['Total_Density'] = df['Density_ID'] + df['Density_FR']
best_state = df.loc[df['Total_Density'].idxmax(), 'State']
```

**What Should Have Been Checked:**
- `print(len(df))` after merge - did we lose states?
- `print(df[df['State'].str.contains('Columbia')])` - is DC present?

---

### Case 6: wildfire-hard-21
**Patterns: Monolithic Block + Wrong Data Source**

| Aspect | Value |
|--------|-------|
| Question | Top 3 states losing most residential property (NOAA data, 2005-2010) |
| Correct Answer | ["California", "Washington", "Idaho"] |
| Code Agent Answer | ["Nevada", "California", "Arizona"] |

**The Monolithic Block (Step 8):**
```python
# Used ZHVI (home values) instead of NOAA wildfire damage
df = pd.read_csv("data/wildfire/input/ZHVI.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Unnamed: 0'])
df_2005 = df[df['Date'].dt.year == 2005]
df_2010 = df[df['Date'].dt.year == 2010]
# ... computed housing price decline, not wildfire property damage
```

**What Should Have Been Checked:**
- Question says "NOAA data" - should have verified using NOAA wildfire files
- ZHVI measures home values (housing crisis), not wildfire damage

---

### Case 7: astronomy-hard-9
**Pattern: Error Cascade (not Monolithic Block)**

This case is different - code agent hit actual errors and gave up, guessing "15" instead of computing "24".

---

### Case 8: biomedical-hard-4
**Pattern: Complex File Parsing (not Monolithic Block)**

Code agent couldn't parse Excel file with metadata rows - returned empty list.

---

### Case 9: astronomy-easy-4
**Pattern: Precision/Rounding (not Monolithic Block)**

Code agent computed correct 11.5 years but expected answer was "11 years" (approximate).

---

### Case 10: environment-hard-10
**Pattern: Incomplete Analysis (not Monolithic Block)**

Code agent explored data but jumped to answer "-0.187" without computing correlation.

---

## Key Insight: Why Dataflow's Modular Design Wins

| Code Agent Approach | Dataflow Approach |
|---------------------|-------------------|
| One code block with 4-8 steps | One operator per step |
| Only final output visible | Each operator output visible |
| Bugs hidden in middle steps | Bugs surface immediately |
| No natural checkpoint | Natural checkpoint per operator |
| Context lost between retries | Operator state persists |

### The Critical Difference

**Code Agent Step 2:**
```python
# 50 lines of code
# ...bug on line 23...
# ...
print(final_result)  # Only this is visible
```

**Dataflow Operators:**
```
[Operator 1] → inspect output ✓
[Operator 2] → inspect output ✓  ← Bug caught here!
[Operator 3] → ...
```

---

## Recommendations for Code Agent Improvement

1. **Break monolithic blocks** into smaller steps with intermediate prints
2. **Always verify filter results** - print value counts after filtering
3. **Check merge results** - verify expected entities are present
4. **Use case-insensitive matching** by default for strings
5. **Print intermediate shapes** - `len(df)` after each transformation
6. **Validate data source** matches question requirements

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Cases | 10 |
| Cases with Monolithic Block Pattern | 6 (60%) |
| Cases with Failed Error Recovery (State Loss) | 2 (20%) |
| Cases with Case Sensitivity Bugs | 3 (30%) |
| Cases with Data Source Issues | 1 (10%) |
| Cases where intermediate verification would have caught bug | 6 (60%) |
| Cases where state persistence would have helped | 2 (20%) |

---

## Key Takeaways

### Two Architectural Advantages of Dataflow:

1. **Modularity** (addresses 60% of failures)
   - Each operator produces visible, inspectable output
   - Bugs surface at specific operators, not hidden in code blocks
   - Natural checkpoints for verification

2. **State Persistence** (addresses 20% of failures)
   - Operator outputs are saved and reusable
   - Errors can be fixed incrementally without restarting
   - Previous progress is never lost

### Recommendations for Code Agent:

1. **Add intermediate prints** after each logical step
2. **Use case-insensitive string matching** by default
3. **Verify filter/merge results** before proceeding
4. **Implement checkpointing** - save intermediate DataFrames
5. **Don't give up after errors** - isolate and retry specific steps
6. **Validate data sources** match question requirements
