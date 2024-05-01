from agents.evaluate import Evalutor
import json
import os

# Initialize evaluator
evaluate_tool = Evalutor()

# Function to load JSON data from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to evaluate a single prediction file
def evaluate_predictions(pred_data, bank_info, loan_info):
    bank_eval = {}
    loan_eval = {}
    for key, data in pred_data.items():
        if key in bank_info:
            result = evaluate_tool(prediction=data, true=bank_info[key])
            bank_eval[key] = result
        elif key in loan_info:
            result = evaluate_tool(prediction=data, true=loan_info[key])
            loan_eval[key] = result
    return bank_eval, loan_eval

# Load ground truth data
bank_info = load_json('json/bank_info.json')
loan_info = load_json('json/loan_info.json')

# Prediction file names
prediction_files = [
    'extract_dbrx.json',
    'extract_llama2.json',
    'extract_llama3_70b.json',
    'extract_llama3_8b.json'
]

# Ensure the output directory exists
output_dir = 'json/eval'
os.makedirs(output_dir, exist_ok=True)

# Process each prediction file
for filename in prediction_files:
    pred_data = load_json(f'json/{filename}')
    bank_eval, loan_eval = evaluate_predictions(pred_data, bank_info, loan_info)

    # Save the evaluation results to the appropriate files
    eval_bank_filename = os.path.join(output_dir, f'bank_eval_{filename}')
    eval_loan_filename = os.path.join(output_dir, f'loan_eval_{filename}')
    with open(eval_bank_filename, 'w') as f:
        json.dump(bank_eval, f)
    with open(eval_loan_filename, 'w') as f:
        json.dump(loan_eval, f)
