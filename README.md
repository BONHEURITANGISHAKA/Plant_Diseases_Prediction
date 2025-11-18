# Plant_Diseases_Prediction
Plant Disease Prediction using Machine Learning
Project Overview
This project focuses on building and evaluating machine learning models to predict the presence of plant disease based on environmental factors such as temperature, humidity, rainfall, and soil pH. The goal is to demonstrate a complete data science workflow, from exploratory data analysis and preprocessing to model training, evaluation, and interpretation. This project showcases proficiency in data analysis, machine learning fundamentals, and effective communication of results.

Dataset
The dataset plant_disease_dataset.csv contains environmental measurements and a target variable indicating the presence (1) or absence (0) of plant disease. It includes the following features:

temperature (float): Average temperature
humidity (float): Average humidity
rainfall (float): Amount of rainfall
soil_pH (float): Soil pH level
disease_present (int): Target variable (0: No Disease, 1: Disease Present)
Methodology
The project followed a structured approach:

Data Loading and Inspection: Loaded the dataset into a Pandas DataFrame and performed initial checks for structure, data types, and missing values using df.info() and df.describe().

Exploratory Data Analysis (EDA): Conducted a thorough EDA to understand the data distribution, relationships, and potential issues:

Visualized the distribution of the target variable (disease_present) to identify class imbalance.
Generated histograms with KDE for numerical features (temperature, humidity, rainfall, soil_pH) to observe their distributions.
Used box plots to examine the relationship between each numerical feature and the disease_present target.
Plotted a correlation matrix heatmap to understand pairwise relationships between features.
Identified potential outliers in numerical features using box plots.
Data Preprocessing: Prepared the data for machine learning models:

Separated features (X) from the target variable (y).
Identified a significant class imbalance in the disease_present variable (approximately 76% 'No Disease' vs. 24% 'Disease Present').
Performed a stratified train-test split (80% train, 20% test) to ensure consistent class distribution across sets.
Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the classes, mitigating bias towards the majority class.
Scaled numerical features using StandardScaler to standardize their range, which is crucial for distance-based and gradient-descent algorithms.
Model Training and Evaluation: Trained and evaluated three different classification models:

K-Nearest Neighbors (KNN) Classifier
Random Forest Classifier
XGBoost Classifier Each model's performance was assessed using key metrics: Accuracy, Precision, Recall, F1-Score, and ROC AUC. Confusion matrices were generated, and ROC curves were plotted for visual comparison.
Feature Importance Analysis: Analyzed feature importance for Random Forest and XGBoost models to understand which environmental factors were most influential in predicting disease presence.

Key Findings
Class Imbalance: The initial dataset exhibited a significant class imbalance, which was effectively addressed using SMOTE during preprocessing.
Model Performance: Both XGBoost Classifier and Random Forest Classifier demonstrated strong and comparable performance, significantly outperforming the K-Nearest Neighbors model.
XGBoost achieved the highest ROC AUC (0.8158), indicating excellent discriminatory power.
Random Forest also performed very well with a high ROC AUC (0.8092) and F1-Score (0.6584).
Feature Importance: soil_pH, rainfall, and humidity were consistently identified as the most important features by both Random Forest and XGBoost models, suggesting their critical role in predicting plant disease. temperature had a comparatively lower impact.
Skills Demonstrated
Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-learn (Logistic Regression, RandomForestClassifier, KNeighborsClassifier, StandardScaler, train_test_split), Imbalanced-learn (SMOTE), XGBoost
Model Evaluation: Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrices
Data Preprocessing: Handling class imbalance, feature scaling
Problem Solving & Critical Thinking: Interpreting model results, understanding limitations, and proposing next steps.
Future Enhancements
Hyperparameter Tuning: Optimize the best-performing models (XGBoost and Random Forest) using techniques like GridSearchCV or RandomizedSearchCV.
Further Feature Engineering: Explore creating new features from existing ones to potentially improve model performance.
Deployment: Consider deploying the best model as an API for real-time predictions.
