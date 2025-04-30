# Transactions Anomaly Detection using Python

A Kaggle-based implementation of an **Isolation Forest**–powered anomaly detection pipeline for financial transactions, replicated and extended from the original tutorial by [The Clever Programmer](https://thecleverprogrammer.com/2023/08/21/anomaly-detection-in-transactions-using-python/).

My notebook lives here on Kaggle:  
https://www.kaggle.com/code/rawsheed91/anomaly-detection-in-transactions-using-python

## 📖 Project Overview

Identifying fraudulent or anomalous transactions is critical for risk management in finance. This project uses an **unsupervised Isolation Forest** model to learn “normal” transaction patterns and flag outliers automatically. Key steps:

1. **Data Ingestion & Exploration**  
2. **Feature Engineering** (e.g. transaction amount, average amount, frequency)  
3. **Model Training** on “clean” data  
4. **Anomaly Scoring & Thresholding**  
5. **Visualization & Analysis** of flagged transactions  

---

## 📂 Dataset

- **Source:** transaction_anomalies_dataset.  
- **Columns used for prediction:**  
  - `Transaction_Amount`  
  - `Average_Transaction_Amount`  
  - `Frequency_of_Transactions` 

---

## 🛠️ Tech Stack

- **Python 3.x**  
- **pandas**, **NumPy** for data handling  
- **scikit-learn** for Isolation Forest  
- **Matplotlib**, **Seaborn**, **Plotly Express** for visualization  
- **Jupyter Notebook** (Kaggle kernel)  

---

## 🚀 How to Run Locally

1. **Clone this repo**  
   ```bash
   git clone https://github.com/OyewoleRasheed/Anomaly-Detection-in-Transactions-using-Python.git
   cd Anomaly-Detection-in-Transactions-using-Python
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   ```

---

## 📈 Key Steps in the Notebook

1. **Load & Inspect Data**  
2. **Compute Features**  
   - Transaction amount  
   - Rolling / average amounts  
   - Frequency counts  
3. **Train Isolation Forest**  
   ```python
   from sklearn.ensemble import IsolationForest
   model = IsolationForest(contamination=0.01, random_state=42)
   model.fit(feature_matrix)
   ```
4. **Score & Label**  
   ```python
   df['anomaly_score'] = model.decision_function(feature_matrix)
   df['is_anomaly']   = model.predict(feature_matrix).map({1: 0, -1: 1})
   ```
5. **Visualize Results**  
   - Distribution of anomaly scores  

---

## 📊 Results & Insights

- **Anomaly rate** ≈ 2% (as set by `contamination`)  
- High-value transactions and sudden frequency bursts are most often flagged.  
- Visualization confirms outliers correspond to unusually large or rare events.
- Added an iteraction user interface to test the model using Gradio
---

## 🤝 Acknowledgements

- **Original tutorial** by The Clever Programmer:  
  https://thecleverprogrammer.com/2023/08/21/anomaly-detection-in-transactions-using-python/  
- Thanks to Kaggle’s free GPU/CPU kernels for rapid iteration and sharing.


---
