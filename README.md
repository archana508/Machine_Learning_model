# House Price Prediction using Machine Learning

## Project Overview
This project focuses on predicting house prices using Machine Learning techniques. The dataset was preprocessed, analyzed, visualized, and used to train multiple regression models for accurate price prediction.

## Objective
To build a Machine Learning model that can predict house prices based on different housing features such as area, bedrooms, bathrooms, parking, furnishing status, and more.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset
Dataset used: Housing Price Dataset

Features include:
- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Air Conditioning
- Furnishing Status
- Preferred Area
- Price (Target Variable)

---

## Steps Performed

### 1. Data Preprocessing
- Loaded dataset using Pandas
- Checked missing values
- Converted categorical values into numerical format

### 2. Exploratory Data Analysis (EDA)
- Displayed dataset information
- Analyzed feature relationships
- Generated visualizations

### 3. Data Visualization
Created:
- Correlation Heatmap
- Actual vs Predicted Graph
- Price Distribution Graph
- Area vs Price Scatter Plot

### 4. Model Building
Implemented:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 5. Model Evaluation
Evaluated models using:
- RMSE (Root Mean Square Error)
- R2 Score

---

## Project Output
The Random Forest model achieved better prediction accuracy compared to other models.

---

## How to Run the Project

### Install Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project
```bash
python house_price_prediction.py
```

---

## Project Structure

```text
House_Price_Prediction/
│
├── house_price_prediction.py
├── Housing.csv
├── README.md
├── requirements.txt
└── screenshots/
```

---

## Learning Outcomes
- Data preprocessing techniques
- Feature engineering
- Data visualization
- Regression model training
- Model evaluation techniques
- Machine Learning workflow
