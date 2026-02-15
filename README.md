# ML_EmailSpamClassifier
An Email Spam Classifier using various ML models

**Problem Statement** 

Email spam detection is a binary classification problem where the objective is to automatically classify emails as Spam or Not Spam based on their content characteristics.
The goal of this assignment is to design, evaluate, and deploy a machine learning–based classification system that can accurately identify spam emails using supervised learning techniques.

**Dataset Descritption**
  1. Dataset Name: Spambase
  2. Source: UCI Machine Learning Repository
  3. Link: https://archive.ics.uci.edu/ml/datasets/Spambase
  4. Number of instances: 4601
  5. Number of features: 57
  6. Target variable:  
     1 → Spam  
     0 → Not Spam  
  8. Total columns: 58 (57 features + 1 label)

**Models Used and Evaluation metrics**  

The following supervised learning models were implemented and evaluated:

1. Logistic Regression    
3. Decision Tree Classifier  
7. Naive Bayes  
8. Random Forest  
9. XGBoost  

<img width="894" height="332" alt="image" src="https://github.com/user-attachments/assets/4a3a42e4-9c03-40bc-9801-9857ebc22ef3" />  
  
  
**Model Performnace**  

<img width="663" height="410" alt="image" src="https://github.com/user-attachments/assets/dadcf8fd-7adb-40c6-9251-5aed0e0b0720" />

Model-Specific Considerations
  1. Feature scaling is applied for:  
    . Logistic Regression  
    . KNN  
  3. Tree-based models are used for feature importance analysis

**Streamlit App Link:** https://mlemailspamclassifier-kandarpprakash.streamlit.app/

<img width="1909" height="869" alt="image" src="https://github.com/user-attachments/assets/b6260877-16a7-47b1-81b8-4c532f139ca3" />



