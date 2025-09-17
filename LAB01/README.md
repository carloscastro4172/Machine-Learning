# Titanic Survival Prediction & EDA

This project provides an **interactive Streamlit app** for exploring the Titanic dataset and predicting passenger survival using a **Decision Tree model** wrapped in a Scikit-learn pipeline.  

The app includes:
- Exploratory Data Analysis (EDA) with filters, metrics, and charts.  
- Interactive visualizations: distributions, boxplots, survival comparisons, correlation matrix.  
- Prediction tab where you can input passenger features and get survival predictions with probabilities.  

---

## Project Structure
- `app.py` → Streamlit app for EDA and predictions.  
- `MachineLe_01.ipynb` → Jupyter Notebook used to preprocess data, train, and export the model.  
- `Titanic-Dataset.csv` → Dataset file (can be replaced with any Titanic-formatted dataset).  
- `titanic_pipeline.pkl` → Serialized Scikit-learn pipeline (Decision Tree + preprocessing).  
- `requirements.txt` → File with dependencies for this project.  

---

## Installation

1. Clone this repository or download the project files.  
2. Create and activate a virtual environment (recommended).  
3. Install all dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the App
1. Place the dataset (`Titanic-Dataset.csv`) and trained model (`titanic_pipeline.pkl`) in the project directory.  
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal (default: `http://localhost:8501/`).  

---

## Features
### 1. EDA Dashboard
- Upload CSV file of Titanic data.  
- Apply filters (Class, Sex, Embarked, Age, Fare, Survived).  
- View:
  - Key Metrics (Survival rate, average age, average fare, passenger count).  
  - Distribution charts for Age and Fare.  
  - Boxplots grouped by categories.  
  - Survival comparisons by Class, Sex, and Embarked.  
  - Correlation heatmap of numerical features.  
  - Key insights auto-generated from the dataset.  

### 2. Prediction
- Input passenger features: Class, Sex, Age, SibSp, Parch, Fare, Embarked.  
- Model predicts:
  - Survival outcome.  
  - Probability of survival / non-survival.  

---

## Example Prediction
**Input:**  
- Class: `1`  
- Sex: `female`  
- Age: `25`  
- SibSp: `0`  
- Parch: `0`  
- Fare: `80.0`  
- Embarked: `C`  

**Output:**  
Passenger survives (Probability: ~92%).  

---

## Notes
- Model can be retrained in `MachineLe_01.ipynb` and re-exported as `titanic_pipeline.pkl`.  
- The app is designed to work with the classic Titanic dataset (like Kaggle).  
- Extendable with other classifiers (Random Forest, Logistic Regression, etc.).  
