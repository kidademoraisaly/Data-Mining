# Airline Customer & Flights Data Mining Project

## Overview
This project explores customer flight behaviour using an airline loyalty dataset.  
It combines customer demographic and enrolment data with flight activity records to identify key behavioural patterns such as seasonality, companion travel, and loyalty value correlations.

The project includes exploratory data analysis (EDA) and geospatial analysis, with both static and interactive visualisations.

---

##  Project Structure

```
DATA-MINING/
│
├── data/
│   ├── DM_AIAI_CustomerDB.csv        # Raw customer data
│   ├── DM_AIAI_FlightsDB.csv         # Raw flight data
│   ├── DM_AIAI_Metadata.csv          # Metadata definitions
│
├── flights_features.csv              # Engineered flight-level features
├── customers_features.csv            # Engineered customer-level features
├── combined_features.csv             # Final dataset (merged customers + flights)
│
├── utils.py                          # Helper functions for EDA visualisations
├── geospatial_analysis.py            # Functions for interactive Plotly mapping
│
├── Group55_EDA.ipynb                 # Exploratory data analysis notebook
├── Group55_geospatial_analysis.ipynb # Geospatial and interactive analysis
│
├── .gitattributes / .gitignore       # Git configuration files
└── README.md                         # Project documentation (this file)
```

---

##  Main Components

### 1. Data Preparation
- Load and clean raw data from customer and flight databases.
- Aggregate flight metrics (number of flights, companion flights, redemption ratios).
---

## 2. Feature Engineering

### Customer-Level Features

| Feature | Description |
|----------|-------------|
| **DaysSinceEnrollment** | Number of days since the customer enrolled in the loyalty programme. Useful to measure tenure and engagement. |
| **DaysSinceCancellation** | Number of days since the customer cancelled their membership. Helps identify inactive customers. |
| **EnrollmentDurationInDays** | Total duration between enrolment and cancellation (how long the customer stayed active). |
| **IsCancelled** | Binary flag (1 = cancelled, 0 = active) indicating current status in the programme. |


---

### Flight Activity Metrics

These metrics are computed only for active months (months where at least one flight was taken).  
They capture **frequency**, **recency**, and **engagement** in customer travel patterns.

| Feature | Description |
|----------|-------------|
| **SumNumFlights** | Total number of flights taken by the customer. |
| **AvgNumFlights** | Average number of flights per active month. |
| **VarStdNumFlights** | Standard deviation of the number of flights across months (flight frequency variability). |
| **NumMonthsWithFlights** | Number of months in which the customer took at least one flight. |
| **PropOfCompanionFlights** | Percentage of flights that included one or more companions. |
| **PropOfRedeem** | Percentage of accumulated points that were redeemed. |
| **MonthsSinceLastFlight** | Number of months since the customer’s most recent flight. |

Additional engineered features:
- **Monthly share metrics**: `Flights_Month_1` … `Flights_Month_12` and `PropCompFlights_Month_1` … `PropCompFlights_Month_12` capture monthly seasonality in flight and companion activity.  
- **Seasonal proportions**: `PropCompFlightsSeason_LowSeasonEarlyYear`, `PropCompFlightsSeason_PeakSeasonSummer`, `PropCompFlightsSeason_Autumn`, `PropCompFlightsSeason_HolidayPeak` summarise broader seasonal trends.

---

### 3. Exploratory Data Analysis (EDA)
Performed in `Group55_EDA.ipynb` using helper functions from `utils.py`:
- Distributions and correlation heatmaps.
- Seasonal and monthly trend analysis.

---

### 4. Geospatial Analysis
Implemented in `geospatial_analysis.py` and `Group55_geospatial_analysis.ipynb`:
- Interactive Plotly maps of customer locations.
- Travel activity clustering and density heatmaps.
- Geographic correlations with CLV.

---

##  Key Insights
- Flight activity shows clear seasonality, peaking during summer** and December.  
- The proportion of companion flights increases in holiday and summer periods.  

---

## Requirements

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn plotly geopandas
```

---

## How to Run
1. Open the project in VS Code or JupyterLab.  
2. Run **`Group55_EDA.ipynb`** for the main exploratory analysis.  
3. Run **`Group55_geospatial_analysis.ipynb`** for interactive mapping.  
4. Use `utils.py` for EDA plots and `geospatial_analysis.py` for Plotly-based geospatial visualisations.

---

## Authors


---
