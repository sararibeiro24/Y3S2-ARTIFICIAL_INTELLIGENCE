from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import logging

project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

from model.src.features.preprocess import split_xy, transform
from model.src.config import TYPE_ENCODING

app = Flask(
    __name__, 
    template_folder='html',  
    static_folder='.',       
    static_url_path=''       
)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)

try:
    model_path = project_dir / "model" / "artifacts" / "model.joblib"
    art = joblib.load(str(model_path))
    print(f"[OK] Model loaded successfully from {model_path}")
    print(f"[OK] Features: {art.feature_names}")
    
    threshold_path = project_dir / "model" / "artifacts" / "high_amount_threshold.joblib"
    if threshold_path.exists():
        HIGH_AMOUNT_THRESHOLD = joblib.load(str(threshold_path))
        print(f"[OK] High amount threshold loaded: {HIGH_AMOUNT_THRESHOLD}")
    else:
        HIGH_AMOUNT_THRESHOLD = None
        print(f"[WARN] high_amount_threshold.joblib not found, is_high_amount will always be 0")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    art = None
    HIGH_AMOUNT_THRESHOLD = None

FEATURES_CONFIG = {
    "step": {
        "label": "Passo (Step)",
        "type": "number",
        "min": 1,
        "max": 743,
        "step": 1,
        "default": 100,
        "help": "Número de passos da simulação (1-743)"
    },
    "type": {
        "label": "Tipo de Transação",
        "type": "select",
        "options": {
            "CASH_IN": "Depósito",
            "CASH_OUT": "Levantamento de Dinheiro",
            "DEBIT": "Débito",
            "PAYMENT": "Pagamento",
            "TRANSFER": "Transferência"
        },
        "default": "TRANSFER",
        "help": "Selecione o tipo de transação"
    },
    "amount": {
        "label": "Montante (€)",
        "type": "number",
        "min": 0,
        "max": 1000000,
        "step": 100,
        "default": 1000,
        "help": "Valor da transação em euros"
    },
    "oldbalanceOrg": {
        "label": "Saldo Anterior - Origem (€)",
        "type": "number",
        "min": 0,
        "max": 1000000,
        "step": 1000,
        "default": 10000,
        "help": "Saldo da conta de origem antes da transação"
    },
    "newbalanceOrig": {
        "label": "Saldo Novo - Origem (€)",
        "type": "number",
        "min": 0,
        "max": 1000000,
        "step": 1000,
        "default": 9000,
        "help": "Saldo da conta de origem após a transação"
    },
    "oldbalanceDest": {
        "label": "Saldo Anterior - Destino (€)",
        "type": "number",
        "min": 0,
        "max": 1000000,
        "step": 1000,
        "default": 5000,
        "help": "Saldo da conta de destino antes da transação"
    },
    "newbalanceDest": {
        "label": "Saldo Novo - Destino (€)",
        "type": "number",
        "min": 0,
        "max": 1000000,
        "step": 1000,
        "default": 6000,
        "help": "Saldo da conta de destino após a transação"
    },
    "hour_of_day": {
        "label": "Hora do Dia",
        "type": "number",
        "min": 0,
        "max": 23,
        "step": 1,
        "default": 12,
        "help": "Hora em que a transação foi realizada (0-23)"
    },
    "day_of_week": {
        "label": "Dia da Semana",
        "type": "select",
        "options": {
            "0": "Segunda-feira",
            "1": "Terça-feira",
            "2": "Quarta-feira",
            "3": "Quinta-feira",
            "4": "Sexta-feira",
            "5": "Sábado",
            "6": "Domingo"
        },
        "default": "1",
        "help": "Dia da semana da transação"
    }
}

def calculate_engineered_features(data):
    engineered = {}

    engineered['errorBalanceOrig'] = (
        data['newbalanceOrig'] + data['amount'] - data['oldbalanceOrg']
    )
    engineered['errorBalanceDest'] = (
        data['oldbalanceDest'] + data['amount'] - data['newbalanceDest']
    )
    engineered['amount_log'] = np.log1p(data['amount'])

    engineered['orig_zero_after']  = int(data['newbalanceOrig'] == 0)
    engineered['dest_zero_before'] = int(data['oldbalanceDest'] == 0)
    engineered['dest_zero_after']  = int(data['newbalanceDest'] == 0)

    if HIGH_AMOUNT_THRESHOLD is not None:
        engineered['is_high_amount'] = int(data['amount'] >= HIGH_AMOUNT_THRESHOLD)
    else:
        engineered['is_high_amount'] = 0

    engineered['is_transfer_or_cashout'] = int(data['type'] in ['TRANSFER', 'CASH_OUT'])

    return engineered

@app.route('/')
def index():
    return render_template('index.html', features_config=FEATURES_CONFIG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if art is None:
            return jsonify({'error': 'Model not loaded'}), 500

        input_data = request.get_json()

        def safe_float(val, default=0.0):
            if val is None or str(val).strip() == '':
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def safe_int(val, default=0):
            if val is None or str(val).strip() == '':
                return default
            try:
                return int(val)
            except ValueError:
                return default

        raw_features = {
            'step':           safe_int(input_data.get('step'), 1),
            'type':           input_data.get('type', 'TRANSFER') if input_data.get('type') else 'TRANSFER',
            'amount':         safe_float(input_data.get('amount')),
            'oldbalanceOrg':  safe_float(input_data.get('oldbalanceOrg')),
            'newbalanceOrig': safe_float(input_data.get('newbalanceOrig')),
            'oldbalanceDest': safe_float(input_data.get('oldbalanceDest')),
            'newbalanceDest': safe_float(input_data.get('newbalanceDest')),
            'hour_of_day':    safe_int(input_data.get('hour_of_day')),
            'day_of_week':    safe_int(input_data.get('day_of_week'))
        }

        engineered = calculate_engineered_features(raw_features)
        raw_features.update(engineered)
        df = pd.DataFrame([raw_features])

        X_transformed = transform(art.preprocessor, df)

        fraud_prob = float(art.model.predict_proba(X_transformed)[0, 1])
        prediction = fraud_prob >= art.threshold

        return jsonify({
            'fraud_probability': round(fraud_prob * 100, 2),
            'is_fraud':          bool(prediction),
            'threshold':         art.threshold * 100,
            'engineered_features': {
                'errorBalanceOrig': round(engineered['errorBalanceOrig'], 2),
                'errorBalanceDest': round(engineered['errorBalanceDest'], 2),
                'amount_log':       round(engineered['amount_log'], 2)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/features-config')
def get_features_config():
    return jsonify(FEATURES_CONFIG)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)