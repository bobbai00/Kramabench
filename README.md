# SEC Data Source

The collected SEC data source currently contain the following sub-datasources:

(All python scripts should be run from the root folder, e.g., `python3 scripts/DERA/CF.py`)

## EDGAR Reports

It contains ~ 200K reports, each containing a few `html` documents, `txt`s, `table`s, and even `image`s.

Currently, sampled 100 reports are in the Github repo, which takes ~ 20 min to download. EDGAR contains a total of ~ 200K reports (which means about 27 days of estimated downloading time?).

```bash
python3 scripts/EDGAR/download_edgar_data.py
python3 scripts/EDGAR/preprocess_filings.py
```

File structure:
```
data/
└── SEC/
    └── EDGAR/
        └── Main/
            ├── <form_type>/
            |   └── <form_id>/
            |       ├── filing.json
            |       ├── filing.pickle
            |       └── *
            └── sampled_filings.pkl
```

## DERA

### CF (Crowdfunding Offerings Data Sets)

It contains quarterly data from 2016 Q2. Mainly tsv `table`s with `html` readme and `json` metadata attached. We convert all tsv files to comma-separated csv files, and extract the shared schema to dataset root folder.

```bash
python3 scripts/DERA/CF.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── CF/
            ├── <year>/
            |   └── <quarter>/
            |       ├── FORM_C_DISCLOSURE.csv
            |       ├── FORM_C_ISSUER_INFORMATION.csv
            |       ├── FORM_C_ISSUER_JURISDICTIONS.csv
            |       ├── FORM_C_ISSUER_SIGNATURE.csv
            |       ├── FORM_C_SIGNATURE.csv
            |       └── FORM_C_SUBMISSION.csv
            ├── cf_metadata.json
            └── cf_readme.html
```

### FS (Financial Statement Data Sets)

It contains quarterly data from 2009 Q1 (but it seems that 2009 Q1 is empty, data starts from 2009 Q2). Mainly tab-separated tables in txt. Unified to csv `table`s. We convert all tsv files to comma-separated csv files, and extract the shared schema to dataset root folder. From 2024 Q3, the schema changed. Not sure if it is a temporary change for the latest month of a permanent change.

**Due to the large full data file size (~25GB), data is NOT uploaded to GitHub.**

```bash
python3 scripts/DERA/FS.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── FS/
            ├── <year>/
            |   └── <quarter>/
            |       ├── num.csv
            |       ├── pre.csv
            |       ├── sub.csv
            |       └── tag.csv
            └── readme.htm
```

### RR (Mutual Fund Prospectus Risk/Return Summary Data Sets)

It contains quarterly data from 2010 Q4. Mainly tsv `table`s with `htm` readme and `json` metadata attached. We convert all tsv files to comma-separated csv files, and extract the shared schema to dataset root folder.

From 2019 Q1 to 2020 Q3, there are slight differences in the metadata files (foreign keys). From 2023 Q3, there is a minor negligible format difference.

**Due to the large full data file size (~10GB), data is NOT uploaded to GitHub.**

```bash
python3 scripts/DERA/RR.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── CF/
            ├── <year>/
            |   └── <quarter>/
            |       ├── cal.csv
            |       ├── lab.csv
            |       ├── num.csv
            |       ├── sub.csv
            |       ├── tag.csv
            |       └── txt.csv
            ├── rr1_metadata.json
            └── readme.htm
```

### TA (Transfer Agent Data Sets)

It contains quarterly data from 2009 Q1. Mainly tsv `table`s. We convert all tsv files to comma-separated csv files.

```bash
python3 scripts/DERA/TA.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── TA/
            ├── <year>/
            |   └── <quarter>/
            |       ├── TA2_FILING.csv
            |       ├── TA_DISCIPLINARY_HIST_DETAILS.csv
            |       ├── TA_SERVICE_COMPANIES.csv
            |       ├── TA_SUBMISSION.csv
            |       ├── TA1_ADDRESS.csv
            |       ├── TA1_CONTROL_ENTITIES.csv
            |       ├── TA1_CORP_PARTNER_DATA.csv
            |       ├── TA1_REGISTRANT.csv
            |       ├── TA2_DB_SEARCH.csv
            |       ├── TA2_SECURITY_HOLDER_ACCOUNTS.csv
            |       ├── TAW_ENTITY.csv
            |       └── TAW_FILING.csv
            ├── ta_metadata.json
            └── ta_readme.html
```

### papers (Staff Papers and Analyses)

It contains yearly data from 2007. Mainly `pdf`s.

```bash
python3 scripts/DERA/papers.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── papers/
            └── <year>/
                └── *.pdf
```

## DATA (SEC Data Library)

### ADV

It contains monthly data from June 2006. Mainly xlsx anc csv `table`s. We convert all xlsx files to comma-separated csv files.

```bash
python3 scripts/DATA/ADV.py
```

File structure:
```
data/
└── SEC/
    └── DERA/
        └── TA/
            ├── <year>/
            |   └── <quarter>/
            |       ├── cal.csv
            |       ├── lab.csv
            |       ├── num.csv
            |       ├── sub.csv
            |       ├── tag.csv
            |       └── txt.csv
            ├── rr1_metadata.json
            └── readme.htm
```

# Queries

Manually created JSON queries are under `quereis/*`. Currently it is classified into `easy`, `medium` and `hard`.

# Update Log

## 2024.12.19

- Added PDF readmes for SEC datasets

- Added text content and header to EDGAR filings

- Added the DERA/papers dataset