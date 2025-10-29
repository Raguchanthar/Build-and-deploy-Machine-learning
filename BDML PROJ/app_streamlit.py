import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------

BEST_MODEL_CANDIDATES = [
	'models/randomforest_pipeline.joblib',
	'models/decisiontree_pipeline.joblib',
	'models/svm_(optional)_pipeline.joblib',
]

COMMON_TARGET_NAMES = ['Churn', 'churn', 'Exited', 'Customer_Churn']


def find_model_path() -> Optional[str]:
	for p in BEST_MODEL_CANDIDATES:
		if os.path.exists(p):
			return p
	# fallback: pick any .joblib in models/
	models_dir = 'models'
	if os.path.isdir(models_dir):
		for f in os.listdir(models_dir):
			if f.endswith('.joblib'):
				return os.path.join(models_dir, f)
	return None


def read_table(uploaded_file, sheet_name=None) -> pd.DataFrame:
	name = uploaded_file.name.lower()
	if name.endswith('.xlsx') or name.endswith('.xls'):
		return pd.read_excel(uploaded_file, sheet_name=sheet_name)
	return pd.read_csv(uploaded_file)


def autodetect_target(df: pd.DataFrame) -> Optional[str]:
	for c in COMMON_TARGET_NAMES:
		if c in df.columns:
			return c
	return None


def evaluate(df: pd.DataFrame, target_col: str, model) -> Tuple[float, float]:
	from sklearn.metrics import accuracy_score, f1_score
	X = df.drop(columns=[target_col])
	y = df[target_col]
	y_pred = model.predict(X)
	acc = accuracy_score(y, y_pred)
	f1 = f1_score(y, y_pred, average='macro')
	return acc, f1


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title='Telecom Churn Predictor', page_icon='📉', layout='wide')
st.title('📉 Telecom Customer Churn Predictor')
st.write('Upload a CSV or Excel file to get churn predictions. If a target column is present, evaluation metrics will be shown as well.')

with st.sidebar:
	st.header('Model')
	model_path = find_model_path()
	if model_path:
		st.success(f'Loaded model: {model_path}')
		model = joblib.load(model_path)
	else:
		st.error('No trained model found in models/. Please run the notebook to train and save a model.')
		model = None

	st.header('Upload Data')
	uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])
	sheet_name = st.text_input('Excel sheet name (optional)', value='')
	if not sheet_name:
		sheet_name = None

if uploaded is None:
	st.info('Awaiting file upload...')
	st.stop()

try:
	df = read_table(uploaded, sheet_name)
except Exception as e:
	st.error(f'Failed to read file: {e}')
	st.stop()

st.subheader('Preview')
st.dataframe(df.head(20))
st.caption(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')

if model is None:
	st.stop()

# Choose target column if present
target_auto = autodetect_target(df)
target_col = st.selectbox('Target column (optional, for evaluation)', options=['<none>'] + list(df.columns), index=(0 if target_auto is None else (list(df.columns).index(target_auto) + 1)))
if target_col == '<none>':
	target_col = None

# Predict / Evaluate
run = st.button('Run Prediction')
if run:
	data_for_pred = df.copy()
	if target_col and target_col in data_for_pred.columns:
		st.write('Target column detected. Running evaluation...')
		try:
			acc, f1 = evaluate(data_for_pred, target_col, model)
			st.metric('Accuracy', f'{acc:.4f}')
			st.metric('F1-macro', f'{f1:.4f}')
		except Exception as e:
			st.error(f'Failed to evaluate: {e}')

	# Predictions
	if target_col and target_col in data_for_pred.columns:
		X = data_for_pred.drop(columns=[target_col])
	else:
		X = data_for_pred

	try:
		y_pred = model.predict(X)
		pred_df = data_for_pred.copy()
		pred_df['prediction'] = y_pred
		st.subheader('Predictions')
		st.dataframe(pred_df.head(50))

		# Download predictions
		csv_buf = io.StringIO()
		pred_df.to_csv(csv_buf, index=False)
		st.download_button('Download Predictions (CSV)', csv_buf.getvalue(), file_name='predictions.csv', mime='text/csv')
	except Exception as e:
		st.error(f'Failed to predict: {e}')


