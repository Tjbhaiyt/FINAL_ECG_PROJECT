# ECG Signal Classifier
Classifies ECG heartbeats using MIT-BIH Arrhythmia Dataset.
Random Forest model achieving 98% accuracy.

## Classes
- Normal (N)
- Supraventricular ectopic beats (SVEB)
- Ventricular ectopic beats (VEB)
- Fusion beats (F)
- Unknown beats (Q)

## Setup
1. pip install -r requirements.txt
2. Run jashu.ipynb to generate ecg_model.joblib
3. python app.py
4. Open http://127.0.0.1:5000