# Heart Failure Prediction Project 🫀

Welcome to the Heart Failure Prediction project! This project uses Artificial Intelligence (AI) to predict the likelihood of heart failure based on a patient's medical data.

## 🚀 Quick Start

To run the analysis, simply execute the following command in your terminal:

```bash
python heart_failure_prediction.py
```

## 📖 How It Works (Step-by-Step)

This script (`heart_failure_prediction.py`) works like a medical team analyzing patient records. Here is the process broken down into simple steps:

### 1. Loading the Data 📂
First, the script reads the `heart.csv` file. This file contains medical records for hundreds of patients, including details like Age, Cholesterol levels, and Chest Pain type.

### 2. "Feature Engineering" (Making the Data Better) 🛠️
Before showing the data to the AI, we improve it by creating new "clues" (features) that might help:
*   **Ratios**: We calculate things like `Cholesterol / Age`. Sometimes, a high cholesterol level is more dangerous for a younger person than an older one, so this ratio helps the AI understand that context.
*   **Squared Values**: We square the `Oldpeak` value (a heart measurement) to help the AI see if extreme values are much more dangerous.

### 3. Translating for the Computer (Preprocessing) 🔢
Computers only understand numbers, not words.
*   **Encoding**: We change words like "Male/Female" or "Asymptomatic" into numbers (0s and 1s).
*   **Scaling**: We adjust all the numbers so they are on a similar scale. For example, `Cholesterol` (e.g., 200) is a much bigger number than `Age` (e.g., 50). Scaling ensures the AI doesn't think Cholesterol is "more important" just because the number is bigger.

### 4. Training the AI Models (The "Doctors") 👨‍⚕️
We train four different AI models. Think of them as four different doctors giving their opinion:
1.  **Logistic Regression**: A standard statistical approach. Good at finding simple relationships.
2.  **Random Forest**: Imagine asking 100 doctors and taking the majority vote. This model builds many "decision trees" to make a prediction.
3.  **XGBoost**: A very smart model that learns from its mistakes. It builds trees one by one, focusing on the hard-to-predict patients.
4.  **SVM (Support Vector Machine)**: This model tries to draw a clear line (or boundary) between healthy and sick patients in a complex multi-dimensional space.

### 5. The "Ensemble" (Teamwork) 🤝
Finally, we combine all four models into an **Ensemble**.
*   Instead of trusting just one model, we let them all vote.
*   This "Team of Doctors" approach usually gives the most accurate result (Accuracy: **~89.13%**).

## 📊 Understanding the Results

When the script finishes, it produces three main things:

1.  **Table 1 (Class Balance)**: Shows how many patients in the dataset have heart disease vs. how many are normal.
2.  **Table 2 (Scorecard)**: Shows the accuracy of each model. Look for the **Ensemble** score to see the best performance.
3.  **Figure 1 (Feature Importance)**: A chart saved as `heart_figure1.png`. It shows which medical factors were most important for the prediction.
    *   *Example*: If `ST_Slope_Up` is at the top, it means that specific heart measurement is the biggest warning sign.

## 📦 Requirements

To run this, you need Python installed with these libraries:
*   `pandas` (for data handling)
*   `scikit-learn` (for the AI models)
*   `xgboost` (for the advanced AI model)
*   `matplotlib` & `seaborn` (for plotting charts)

---
*Created by Antigravity*
