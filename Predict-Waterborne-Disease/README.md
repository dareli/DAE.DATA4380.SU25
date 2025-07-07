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
![Corr_heatmap](num_corr.png) 

- Boxplots & scatterplots below comparing disease counts to features showed no clear trends (scatter was filled)
![Boxplot_diseases](diseases_box.png)
![Scatterplot_features](features_scatter.png)

- Histograms of numeric features indicated no significant outliers, as confirmed by the IQR method.
![numerical_hists](num_hists.png)

## **Problem Formulation**
- Input: 20 numeric + 4 categorical features (after preprocessing and feature engineering)
- Output: Regression targets for three diseases
- **Models Used:**
  - Baseline Random Forest Regressor
  - Random Forest
  - XGBoost (with hyperparameter tuning)
  - Ridge Regression
  - LightGBM
  - CatBoost
-  **Loss/Scoring:**
  - MAE
  - RMSE
  - R²

## **Feature Engineering**
- Created domain-motivated interactions:
  - GDP x Healthcare Index
  - Sanitation Gap (100% - Sanitation Coverage)
  - CleanWater x Urbanization
  - Infant Mortality per GDP
- Pollution Index (sum of selected contamination metrics)

## **Training**
- **Environment:** Python, Jupyter Notebook
- **Libraries used:** scikit-learn, xgboost, lightgbm, catboost, pandas, numpy, seaborn, matplotlib
- **Training time:** about a few seconds per model
- **Cross-validation:** 3-fold for tuning hyperparameters
- No overfitting was observed, but the data showed a generally weak signal.
- **Challenges:** Very weak correlations in data made modeling difficult, leading to consistently poor R² scores even after tuning and feature engineering.

## **Performance Comparison** 
- **Key metrics:**
  - MAE (Mean Absolute Error): lower is better
  - RMSE (Root Mean Squared Error): lower is better
  - R²: closer to 1 is better; negative indicates worse than baseline mean prediction

- Baseline Regressor Results Below:
  

## **Conclusions** 
All models performed poorly, showing negative R² values for all targets. This indicates that the water quality and socioeconomic indicators in the dataset do not have predictive power for disease counts at this scale. It is likely that important factors influencing disease burden are missing, such as vaccination rates, local outbreak history, and detailed sanitation infrastructure.

## **Future Work**
- Try classification by grouping case counts into low, medium, and high risk to identify broader patterns that may be easier to predict.
- Improve the dataset by adding more detailed local water quality and health features.
- Enhance feature engineering by testing more interactions, or reductions.

## **How to Reproduce Results** 
- **Overview of Files in Repository:**
  - PWBD_Feasability.ipynb: Exploratory data analysis (EDA) and missing value handling.
  - df_baseline.csv: Cleaned dataset with missing values handled, ready for modeling.
  - PWBD_Prototype_ML.ipynb: Baseline modeling, hyperparameter tuning, feature engineering, and machine learning.
- **Software Setup**
- **Visualization:**
  - matplotlib
  - seaborn
- **Data Handling & Preprocessing:**
  - pandas
  - numpy
  - scipy (for stats and skewness)
  - scikit-learn:
    - train_test_split, Pipeline, ColumnTransformer
    - OneHotEncoder, MinMaxScaler
    - RandomizedSearchCV
- **Machine Learning Models:**
  - scikit-learn regressors:
    - RandomForestRegressor, Ridge, MultiOutputRegressor
  - xgboost
  - lightgbm
  - catboost
  - scikit-learn metrics:
    - mean_absolute_error, mean_squared_error, r2_score

## **Data**
- **Download from: Kaggle Dataset Link**
- Features: Mix of water quality indicators, socioeconomic data, and categorical variables
- Target variables: Diarrheal, Cholera, and Typhoid cases per 100,000 people
- **Preprocessing:**
  - Checked for missing values, duplicates, and outliers
  - Encoded categorical variables and scaled numerics
  - Created new domain-informed interaction features (e.g., pollution index)
- **Training**
  - Split data into training and test sets
  - Built pipelines using scikit-learn
  - Trained multiple regression models: Random Forest, Ridge, XGBoost, LightGBM, CatBoost
  - Performed hyperparameter tuning with RandomizedSearchCV (XGBoost)
  - Tried engineered features to improve results
  - Challenge: Extremely weak correlations and noisy targets led to poor model performance (negative or near-zero R²)
- **Performance Evaluation**:
  - Evaluated models using MAE, RMSE, and R²
  -  Compared results across all models and all 3 disease targets
  -  Created summary tables and visualizations for model performance
  -  Found that no model significantly outperformed others due to weak data relationships

## **Citations**


