Live Website: https://air-craft-predictor-1.onrender.com/


Done by Lokeswaran S

# Hybrid Ensemble Models for Aircraft Accident Severity Classification

Aircraft accident analysis plays a crucial role in improving aviation safety by identifying factors that contribute to varying levels of severity. This project presents a comprehensive machine learning framework to classify aircraft accident severity using a variety of models, ranging from traditional classifiers to advanced hybrid ensemble techniques.

The dataset, preprocessed and visualized with extensive exploratory data analysis (EDA), was used to evaluate the performance of multiple supervised learning models including Ordinal Logistic Regression (OLR), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Naïve Bayes, Random Forest, AdaBoost, and XGBoost.

To enhance predictive accuracy and robustness, hybrid models such as Voting Classifier and Stacking Classifier were implemented. These ensembles combine the strengths of diverse learners including Multi-Layer Perceptron (MLP), tree-based models, and gradient boosting. The final stacked model integrates MLP, Random Forest, and XGBoost as base learners with Logistic Regression as the meta-learner.

The results show that hybrid ensemble models outperform individual classifiers in terms of classification accuracy and generalizability. The project provides a scalable and interpretable approach for classifying aircraft accident severity levels, offering potential support for safety analysis and decision-making in the aviation industry.

## Deploy the Streamlit App

This repository is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push the repository to GitHub, including the `models/` directory.
2. In Streamlit Community Cloud, select **Create app** and choose this repository and branch.
3. Set the main file path to `app/streamlit_app.py`.
4. Deploy the app. Streamlit will install dependencies from `requirements.txt`.

Run it locally with:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

