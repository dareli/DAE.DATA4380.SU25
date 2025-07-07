![UTA-DataScience-Logo](https://github.com/dareli/DATA3402.Spring.2024/assets/123596270/0cb941d4-8a3b-4382-9dd0-22c28edbb8a5)

# **DATA 4380 - Tabular Project: Predicting Waterborne Disease Counts Using Global Water Quality and Socioeconomic Indicators**
This repository holds an attempt to predict waterborne disease incidence (diarrheal, cholera, typhoid) using global water quality and socioeconomic data from the Kaggle dataset "Water Pollution and Disease".

## **Overview** 
- **Project Goal:** The goal is to predict the number of waterborne disease cases (diarrhea, cholera, and typhoid) per 100,000 people at the national level, using 24 features that include water quality measurements and socioeconomic indicators from various countries and regions.
- **Approach:** The problem set up as a supervised regression task and used tabular machine learning methods like Random Forest, XGBoost, Ridge, CatBoost, and LightGBM. We performed preprocessing, feature engineering, and compared models to assess their predictive power on the dataset.
- **Brief Summary of the Performance:** All models consistently produced negative R² scores, which means they had very weak predictive power. The best model (Ridge) had an MAE of about 128 for diarrheal cases per 100,000 people, indicating that the dataset as it is now doesn't allow for reliable predictions.

## **Summary of Work Done**
- **About the Data**
  - Type: Tabular CSV file
  - Input Features (24 total):
    - 20 numerical (e.g. pH level, rainfall, GDP per capita)
    - 4 categorical (Country, Region, Water Source Type, Water Treatment Method)
  - Output:
    - Diarrheal Cases per 100,000 people
    - Cholera Cases per 100,000 people
    - Typhoid Cases per 100,000 people
  - Size: 3000 rows, 24 columns
  - Split:
    - Training: 80%
    - Testing: 20%
   
## **Preprocessing & Clean up**
- Categorical encoding using OneHotEncoder for Water Source Type, Water Treatment Method, Country, and Region.
- Filled in missing values for Water Treatment Method by adding an "Unknown" category.
- Applied standard scaling to numerical features.
- Created interaction features based on domain knowledge.
- Managed missing values.
- Checked feature skew and found all to be low or acceptable.

## **Data Visualization**
- The Correlation heatmap below revealed very weak relationships, with a maximum correlation of about 0.04
![Correlation Matrix](./Documents/plots/correlation_matrix_numerical_columns.png)

- Boxplots & scatterplots below comparing disease counts to features showed no clear trends (scatter was filled)
![Boxplots of Disease Cases](./Documents/plots/boxplots_disease_cases.png)
![Scatter Plots of Targets vs Features](./Documents/plots/scatter_plots_targets_vs_features.png)

- Histograms of numeric features indicated no significant outliers, as confirmed by the IQR method.
![Histograms of Numerical Features](./Documents/plots/histograms_numerical_features.png)
