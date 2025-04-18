# Copper Industry Price Prediction & Lead Classification

This project aims to solve two major challenges faced by the copper industry:

- **Predicting Selling Prices** of copper products based on transaction data.
- **Classifying Sales Leads** as WON or LOST based on customer and transaction features.

We leverage **Machine Learning Regression and Classification Models**, supported by a **Streamlit Web App**, to make data-driven, interactive predictions.

---

## 🗂️ Project Features

- **Data Preprocessing:** Handling missing values, treating outliers, and transforming skewed data.
- **Exploratory Data Analysis (EDA):** Visualizing distributions, outliers, and correlations.
- **Feature Engineering:** Encoding categorical variables, dropping irrelevant features, and creating new informative features.

**ML Models:**

- **Regression:** Predict `selling_price`
- **Classification:** Predict `status (WON/LOST)`

**Streamlit Web App:**

- Choose between **Regression** or **Classification** task
- Enter input values for each feature
- Predict real-time results and display on the page

---

## 📊 Dataset Description

The dataset contains the following columns:

- `id`
- `item_date`
- `quantity tons`
- `customer`
- `country`
- `status` (WON/LOST/Draft etc.)
- `item type`
- `application`
- `thickness`
- `width`
- `material_ref`
- `product_ref`
- `delivery date`
- `selling_price`

---

## 🚀 Tech Stack

- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **Streamlit**

---

## 📈 Workflow

### 1️⃣ Data Understanding & Cleaning
- Convert invalid `material_ref` starting with `00000` to null  
- Handle missing values using mean/median/mode  
- Remove or treat outliers using IQR or Isolation Forest  
- Check and correct skewness using log/BoxCox transformations  

### 2️⃣ EDA & Feature Engineering
- Visualize distributions and outliers using boxplots, distplots, and violinplots  
- Encode categorical columns  
- Drop highly correlated columns  

### 3️⃣ Model Building
- **Split Data:** Train/Test Split  
- **Models Used:**
  - **Regression:** `DecisionTreeRegressor`, `RandomForestRegressor`, `XGBRegressor`
  - **Classification:** `ExtraTreesClassifier`, `XGBClassifier`, `LogisticRegression`
- **Hyperparameter Tuning:** `GridSearchCV` / `RandomizedSearchCV`  

### 4️⃣ Deployment
Create a **Streamlit App** for:

- Task Selection (**Regression** or **Classification**)
- Input Fields for each feature
- Display predicted output
- Pickle (for model serialization)

---

## 📑 How to Run the Project

**Clone the Repository**
```bash
git clone https://github.com/yourusername/copper-industry-modeling.git
cd copper-industry-modeling

Install Required Libraries:
pip install -r requirements.txt

Run the Streamlit App:
streamlit run app.py


