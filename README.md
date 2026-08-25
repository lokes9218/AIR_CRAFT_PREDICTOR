# ✈️ Hybrid Ensemble Models for Aircraft Accident Severity Classification

**Done by:** Lokeswaran S

**Live Website:** https://air-craft-predictor-1.onrender.com/

---

## 📌 Project Overview

Aircraft accident analysis is important for understanding the factors that contribute to different levels of accident severity. Accurate classification of accident severity can support aviation safety analysis, risk assessment, and better decision-making.

This project develops a **machine learning-based aircraft accident severity classification system** that predicts the severity level of an aircraft accident using historical accident-related data.

The project evaluates multiple supervised machine learning algorithms, compares their performance, and combines the strengths of different models using **hybrid ensemble learning techniques**.

The final system is deployed as an interactive web application where users can provide accident-related information and obtain a predicted severity classification.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze aircraft accident data using Exploratory Data Analysis (EDA).
* Preprocess and transform the dataset for machine learning.
* Identify important patterns and relationships between accident-related features.
* Train and evaluate multiple classification algorithms.
* Compare individual machine learning models based on their performance.
* Implement hybrid ensemble learning techniques.
* Improve prediction performance using Voting and Stacking Classifiers.
* Deploy the final machine learning model as a web application.

---

## 📊 Dataset

The project uses aircraft accident-related data containing information about different factors associated with aircraft accidents.

The dataset is processed before training the machine learning models.

### Data Processing Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Handling Missing Values
     ↓
Feature Transformation
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Data Standardization
     ↓
Train-Test Split
     ↓
Model Training
```

---

## 🔍 Exploratory Data Analysis

Extensive Exploratory Data Analysis was performed to understand the dataset and identify important patterns.

The analysis includes:

* Distribution of accident severity levels
* Feature distributions
* Missing-value analysis
* Correlation analysis
* Relationship between important features
* Outlier analysis
* Class distribution visualization
* Feature-level comparisons

EDA helped understand the structure of the dataset before applying machine learning algorithms.

---

# 🤖 Machine Learning Models

Multiple supervised learning algorithms were implemented and evaluated.

### Individual Models

1. **Ordinal Logistic Regression (OLR)**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Decision Tree**
5. **Naïve Bayes**
6. **Random Forest**
7. **AdaBoost**
8. **XGBoost**

These models provide different approaches to solving the classification problem.

For example:

* Logistic Regression provides a simple linear classification approach.
* KNN makes predictions based on neighboring observations.
* SVM identifies an optimal decision boundary.
* Decision Trees use rule-based feature splitting.
* Random Forest combines multiple decision trees.
* AdaBoost combines weak learners sequentially.
* XGBoost uses gradient boosting for powerful predictive performance.

---

# 🔥 Hybrid Ensemble Learning

To improve the performance of individual models, hybrid ensemble techniques were implemented.

Ensemble learning combines multiple machine learning models so that their collective prediction can be more robust than relying on a single classifier.

## 🗳️ Voting Classifier

The Voting Classifier combines predictions from multiple base models.

```text
        Model 1
           ↓
        Model 2 ──→ Voting ──→ Final Prediction
           ↓
        Model 3
```

Different models contribute to the final prediction, allowing the ensemble to utilize the strengths of multiple learning algorithms.

---

# 🧠 Stacking Classifier

The final hybrid approach uses a **Stacking Classifier**.

The stacking architecture consists of:

### Base Learners

* Multi-Layer Perceptron (MLP)
* Random Forest
* XGBoost

### Meta Learner

* Logistic Regression

Architecture:

```text
                Input Features
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        MLP     Random Forest  XGBoost
          │          │          │
          └──────────┼──────────┘
                     ↓
              Base Predictions
                     ↓
            Logistic Regression
              Meta Learner
                     ↓
             Final Prediction
```

The base models learn different patterns from the input data. Their predictions are then provided to the Logistic Regression meta-learner, which learns how to combine those predictions to produce the final classification.

---

# 📈 Model Evaluation

The machine learning models are evaluated using standard classification metrics.

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive instances are actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual positive instances are correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

Provides a balance between precision and recall.

```text
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

These metrics help compare the performance and generalization ability of the implemented models.

---

# 🏆 Final Model

The project focuses on the **Stacking Classifier** as the final hybrid ensemble approach.

The final architecture combines:

```text
MLP
 │
 ├──────────────┐
 │              │
Random Forest   │
 │              ├──→ Logistic Regression
XGBoost         │     Meta Learner
 │              │
 └──────────────┘
        ↓
 Final Severity Prediction
```

By combining neural-network-based learning, bagging-based tree models, and gradient boosting, the stacking approach can capture different types of patterns present in the dataset.

The hybrid ensemble approach provides a robust framework for aircraft accident severity classification.

---

# 🌐 Web Application

The trained machine learning model is integrated into an interactive web application.

### Live Application

**Aircraft Accident Severity Predictor**

https://air-craft-predictor-1.onrender.com/

The application allows users to enter the required aircraft accident information and receive a predicted accident severity classification.

### Application Workflow

```text
User Input
    ↓
Input Validation
    ↓
Data Preprocessing
    ↓
Trained ML Model
    ↓
Severity Prediction
    ↓
Result Display
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit

### Development & Deployment

* Git
* GitHub
* Render

---

# 📁 Project Structure

```text
Aircraft-Accident-Severity-Classification/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── trained_model_files
│   └── preprocessing_files
│
├── data/
│   └── dataset_files
│
├── notebooks/
│   └── model_training_and_analysis.ipynb
│
├── requirements.txt
│
├── README.md
│
└── other project files
```

> The exact file structure may vary depending on the files included in the repository.

---

# 🚀 Running the Project Locally

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

## 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Run the Streamlit Application

```bash
python -m streamlit run app/streamlit_app.py
```

The application will start locally and can be accessed through the URL provided by Streamlit.

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud** or other cloud platforms that support Python/Streamlit applications.

### Deployment Steps

1. Push the complete project to GitHub.
2. Make sure the `models/` directory is included.
3. Open Streamlit Community Cloud.
4. Select **Create App**.
5. Select the GitHub repository and branch.
6. Set the main file path:

```text
app/streamlit_app.py
```

7. Deploy the application.
8. Streamlit installs the required packages using:

```text
requirements.txt
```

---

# 🔄 Complete Project Workflow

```text
                Aircraft Accident Dataset
                          │
                          ↓
                   Data Preprocessing
                          │
                          ↓
                     Data Cleaning
                          │
                          ↓
                         EDA
                          │
                          ↓
                  Feature Engineering
                          │
                          ↓
                  Data Standardization
                          │
                          ↓
                    Model Training
                          │
        ┌─────────────────┼──────────────────┐
        ↓                 ↓                  ↓
   Individual         Ensemble Models     Hybrid Models
     Models
        │                 │                  │
        ↓                 ↓                  ↓
 OLR, KNN, SVM       Voting Classifier   Stacking Classifier
 Decision Tree
 Naive Bayes
 Random Forest
 AdaBoost
 XGBoost
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ↓
                  Model Comparison
                          │
                          ↓
                    Final Model
                          │
                          ↓
                  Streamlit Web App
                          │
                          ↓
                   Live Prediction
```

---

# 💡 Key Highlights

* Implemented multiple supervised machine learning algorithms.
* Compared traditional classification models with ensemble approaches.
* Implemented Voting Classifier for combined model predictions.
* Implemented Stacking Classifier using multiple base learners.
* Used **MLP, Random Forest, and XGBoost** as base learners.
* Used **Logistic Regression** as the meta-learner.
* Performed EDA and data preprocessing before model training.
* Evaluated models using accuracy, precision, recall, and F1-score.
* Developed an interactive Streamlit prediction application.
* Deployed the application online for real-time predictions.

---

# 🎓 Learning Outcomes

Through this project, the following concepts were explored:

* Supervised Machine Learning
* Multiclass/ordinal classification
* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Model comparison
* Ensemble learning
* Voting Classifier
* Stacking Classifier
* Hyperparameter-based model optimization
* Model evaluation
* Machine Learning deployment
* Streamlit application development

---

# 🔮 Future Enhancements

The project can be further improved by:

* Adding more recent aircraft accident data.
* Performing advanced feature engineering.
* Applying systematic hyperparameter optimization.
* Adding explainable AI techniques such as SHAP.
* Providing probability scores for each severity class.
* Adding interactive visualizations to the web application.
* Implementing model monitoring after deployment.
* Exploring deep learning approaches for further improvement.
* Integrating additional aviation safety datasets.

---

# 👨‍💻 Author

**Lokeswaran S**

Computer Science Engineering Student

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
