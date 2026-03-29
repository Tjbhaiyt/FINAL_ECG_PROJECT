import os
import gdown

MODEL_PATH = "ecg_model.joblib"
FILE_ID = "1A-Cu1Sdyv106QbTEkZ1enGAsbtPgC8A_"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Now load your model normally
import joblib
model = joblib.load(MODEL_PATH)
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import io
from scipy.interpolate import interp1d


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
try:
    model = joblib.load('ecg_model.joblib')
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model does not support predict_proba.")
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ 'ecg_model.joblib' not found. Run jashu.ipynb first.")
    model = None

EXPECTED_FEATURES = 187

class_mapping = {
    0: 'Normal (N)',
    1: 'Supraventricular ectopic beats (SVEB)',
    2: 'Ventricular ectopic beats (VEB)',
    3: 'Fusion beats (F)',
    4: 'Unknown beats (Q)'
}

class_colors = {
    'Normal (N)':                            '#22c55e',
    'Supraventricular ectopic beats (SVEB)': '#f59e0b',
    'Ventricular ectopic beats (VEB)':       '#ef4444',
    'Fusion beats (F)':                      '#8b5cf6',
    'Unknown beats (Q)':                     '#6b7280'
}

class_info = {
    'Normal (N)':                            'Heart rhythm appears normal. No arrhythmia detected.',
    'Supraventricular ectopic beats (SVEB)': 'Abnormal beat originating above the ventricles. Monitor advised.',
    'Ventricular ectopic beats (VEB)':       'Abnormal beat originating in the ventricles. Consult a cardiologist.',
    'Fusion beats (F)':                      'Combination of normal and ventricular beat. Further evaluation needed.',
    'Unknown beats (Q)':                     'Unclassifiable beat. Further clinical evaluation recommended.'
}


# ─────────────────────────────────────────────
# Resample ANY length → exactly 187 via interpolation
# ─────────────────────────────────────────────
def resample_to_187(values):
    """
    Linearly interpolates input signal of any length to exactly 187 points.
    This preserves waveform shape — much more accurate than zero-padding.
    Works for 50, 100, 150, 200, 300... any count.
    """
    n = len(values)
    if n == EXPECTED_FEATURES:
        return np.array(values, dtype=float), None

    x_input  = np.linspace(0, 1, n)
    x_output = np.linspace(0, 1, EXPECTED_FEATURES)
    interpolator = interp1d(x_input, values, kind='linear')
    return interpolator(x_output), n


# ─────────────────────────────────────────────
# Normalize: min-max scale to [0, 1]
# ─────────────────────────────────────────────
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return arr  # flat signal — return as-is
    return (arr - mn) / (mx - mn)


# ─────────────────────────────────────────────
# Master predict function
# ─────────────────────────────────────────────
def parse_and_predict(features_raw):
    warnings = []
    count = len(features_raw)

    if count == 0:
        return {'error': 'No feature values found in input.'}

    if count < 2:
        return {'error': 'At least 2 values are required to classify a signal.'}

    # Strip label column if 188 values given
    if count == EXPECTED_FEATURES + 1:
        warnings.append(
            "188 values detected — the last value (label column) was automatically removed."
        )
        features_raw = features_raw[:EXPECTED_FEATURES]
        count = EXPECTED_FEATURES

    # Resample to 187 using interpolation
    resampled, original_count = resample_to_187(features_raw)

    if original_count is not None:
        direction = "stretched" if original_count < EXPECTED_FEATURES else "compressed"
        warnings.append(
            f"{original_count} values provided — signal was {direction} to "
            f"{EXPECTED_FEATURES} points using linear interpolation for accurate classification."
        )

    # Normalize to [0, 1]
    normalized = normalize(resampled)

    if not model:
        return {'error': 'Model is not loaded. Please check server logs.'}

    sample = normalized.reshape(1, -1)
    pred_numeric  = int(model.predict(sample)[0])
    pred_label    = class_mapping[pred_numeric]
    probabilities = model.predict_proba(sample)[0]

    confidences = {
        class_mapping[i]: {
            'percent': f"{prob * 100:.2f}",
            'value':   round(prob * 100, 2),
            'color':   class_colors[class_mapping[i]]
        }
        for i, prob in enumerate(probabilities)
    }

    return {
        'prediction':    pred_label,
        'color':         class_colors[pred_label],
        'info':          class_info[pred_label],
        'confidences':   confidences,
        'warnings':      warnings,
        'feature_count': count,
        'resampled_to':  EXPECTED_FEATURES
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict/manual', methods=['POST'])
def predict_manual():
    ecg_string = request.form.get('ecg_data', '').strip()
    if not ecg_string:
        return jsonify({'error': 'No data entered. Please paste your ECG feature values.'})
    try:
        features = [
            float(v.strip())
            for v in ecg_string.replace('\n', ',').split(',')
            if v.strip() != ''
        ]
    except ValueError:
        return jsonify({'error': 'Invalid input. Only comma-separated numbers are accepted.'})

    if len(features) == 0:
        return jsonify({'error': 'No valid numeric values found.'})

    result = parse_and_predict(features)
    return jsonify(result)


@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a valid .csv file.'})

    try:
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content), header=None)

        if df.empty:
            return jsonify({'error': 'The uploaded CSV file is empty.'})

        rows, cols = df.shape

        # Smart format detection:
        # Vertical  → N rows × 1 col  (one value per line)
        # Horizontal→ 1 row  × N cols (all values in one row)
        # Multi-row → use first row
        if cols == 1:
            features = df[0].dropna().tolist()
            csv_fmt  = f"Vertical CSV ({rows} rows × 1 col) — all rows read as one sample."
        else:
            features = df.iloc[0].dropna().tolist()
            csv_fmt  = f"Horizontal CSV ({rows} rows × {cols} cols) — row 1 used."

        features = [float(v) for v in features]
        result   = parse_and_predict(features)
        result['csv_info'] = csv_fmt
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)


    """ app done. To run:
    1. Ensure jashu.ipynb has been run to create 'ecg_model.joblib'.
    2. Install dependencies: pip install flask joblib numpy pandas scipy gdown
    3. Run this script: python app.py
    4. Open http://"""
    