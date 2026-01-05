🧠 Task 4: Responsible AI – Bias & Fairness Analysis
📌 Project Overview

This project focuses on Responsible AI principles, specifically bias detection and fairness analysis in a machine learning model.
We use a healthcare dataset to analyze whether model predictions are biased toward a sensitive attribute (sex).

📂 Folder Structure
Alfido_Task4_Responsible_AI/
│
├── task4_responsible_ai.ipynb
├── README.md
└── dataset.csv

🛠️ Technologies Used

Python 3

Jupyter Notebook

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

📊 Dataset Description

The dataset contains patient health records with attributes such as:

Age

Sex

Blood pressure

Cholesterol

Chest pain type

Heart disease outcome (target)

Sensitive Attribute Used for Fairness Analysis:

sex (Male / Female)

⚙️ Steps Performed
1️⃣ Data Preprocessing

Handled missing values

One-hot encoded categorical features

Feature scaling using StandardScaler

2️⃣ Model Training

Trained a classification model

Generated predictions on test data

3️⃣ Fairness Analysis

Converted sex into encoded form (sex_Male)

Compared positive prediction rates across groups

fairness_df.groupby("sex_Male")["prediction"].mean()

4️⃣ Bias Evaluation Result
Group	Positive Prediction Rate
Female (False)	0.257
Male (True)	0.651

📌 Observation:
The model predicts positive outcomes much more frequently for males, indicating potential gender bias.

⚖️ Responsible AI Considerations

Bias detection using group-wise comparison

Transparency through clear metrics

Awareness of fairness risks in healthcare AI

Encourages ethical model evaluation

✅ Conclusion

This project demonstrates how machine learning models can unintentionally introduce bias and why fairness analysis is essential before real-world deployment—especially in sensitive domains like healthcare.
