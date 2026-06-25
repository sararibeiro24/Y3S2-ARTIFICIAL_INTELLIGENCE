const form = document.getElementById('predictionForm');
const amountInput = document.getElementById('amount');
const amountRange = document.getElementById('amountRange');
const resultsContent = document.getElementById('resultsContent');
const welcomeMessage = document.getElementById('welcomeMessage');
const loadingSpinner = document.getElementById('loadingSpinner');

amountInput.addEventListener('input', (e) => {
    if (e.target.value === '') {
        e.target.value = 0;
    }
    amountRange.value = e.target.value;
    makePrediction();
});

amountRange.addEventListener('input', (e) => {
    amountInput.value = e.target.value;
    makePrediction();
});

form.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', (e) => {
        if (e.target.value === '') {
            e.target.value = 0;
            makePrediction();
        }
    });
});

form.addEventListener('change', makePrediction);
form.addEventListener('input', makePrediction);

window.addEventListener('load', () => {
    makePrediction();
});

async function makePrediction() {
    try {
        const typeSelect = document.getElementById('type');
        if (!typeSelect.value) {
            return; 
        }

        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        showLoading();
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message);
    }
}

function displayResults(result) {
    welcomeMessage.style.display = 'none';
    loadingSpinner.style.display = 'none';
    resultsContent.style.display = 'block';

    const fraudProb = result.fraud_probability;
    const isFraud = result.is_fraud;
    const resultBadge = document.getElementById('resultBadge');
    const badgeTitle = resultBadge.querySelector('.badge-title');
    const badgeSubtitle = resultBadge.querySelector('.badge-subtitle');

    if (isFraud) {
        resultBadge.classList.add('fraud');
        badgeTitle.textContent = 'Fraude Detectada';
        badgeSubtitle.textContent = 'Risco elevado';
    } else {
        resultBadge.classList.remove('fraud');
        badgeTitle.textContent = 'Transação Legítima';
        badgeSubtitle.textContent = 'Risco baixo';
    }

    const probabilityValue = document.getElementById('probabilityValue');
    const progressFill = document.getElementById('progressFill');

    probabilityValue.textContent = fraudProb.toFixed(1) + '%';
    progressFill.style.width = fraudProb + '%';

    const engineered = result.engineered_features;
    document.getElementById('errorBalanceOrig').textContent = engineered.errorBalanceOrig.toFixed(2);
    document.getElementById('errorBalanceDest').textContent = engineered.errorBalanceDest.toFixed(2);
    document.getElementById('amount_log').textContent = engineered.amount_log.toFixed(2);
}

function showLoading() {
    welcomeMessage.style.display = 'none';
    resultsContent.style.display = 'none';
    loadingSpinner.style.display = 'block';
}

function showError(message) {
    welcomeMessage.style.display = 'block';
    resultsContent.style.display = 'none';
    loadingSpinner.style.display = 'none';
    welcomeMessage.innerHTML = `
        <p>Erro: ${message}</p>
        <p style="font-size: 0.85rem; margin-top: 10px;">Verifique se o modelo está carregado corretamente.</p>
    `;
}

function formatCurrency(value) {
    return new Intl.NumberFormat('pt-PT', {
        style: 'currency',
        currency: 'EUR'
    }).format(value);
}

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        makePrediction();
    }
});

document.querySelectorAll('input, select').forEach(element => {
    element.addEventListener('focus', function() {
        this.parentElement.classList.add('focused');
    });
    
    element.addEventListener('blur', function() {
        this.parentElement.classList.remove('focused');
    });
});

console.log('Detector de Fraude inicializado com sucesso');