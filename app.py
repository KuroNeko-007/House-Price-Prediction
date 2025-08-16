from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from src.pipeline.predict_pipeline import CustomData, PredictPipeline, FileUploadData
import io
from datetime import datetime
import uuid


application = Flask(__name__)
app = application

# Configuration
app.config['SECRET_KEY'] = 'ml-prediction-app-secret-key-2024'  # Change this in production!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Create upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## Route for home page
@app.route('/')
def index():
    return render_template('index.html') 

## Route for file upload and prediction
@app.route('/predict_file', methods=['GET', 'POST'])
def predict_file():
    if request.method == 'GET':
        # Read data description from artifacts/Description.txt and pass to template
        try:
            desc_path = os.path.join('artifacts', 'Description.txt')
            data_description = None
            if os.path.exists(desc_path):
                with open(desc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data_description = f.read()
        except Exception:
            data_description = None
        return render_template('upload.html', data_description=data_description)
    else:
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected')
                return redirect(request.url)
            
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
            
            # Check if file type is allowed
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload CSV or Excel files only.')
                return redirect(request.url)
            
            print(f"Processing file: {file.filename}")
            
            # Process the uploaded file
            file_handler = FileUploadData(file)
            df = file_handler.process_uploaded_file()
            
            print(f"File loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            

            df_processed = df.drop(columns=['Id'], errors='ignore')
            print(f"Processed data shape: {df_processed.shape}")
            print(f"Processed columns: {list(df_processed.columns)}")
            
            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            
            # Get predictions
            print("Starting predictions...")
            predictions = predict_pipeline.predict(df_processed)
            
            # Add predictions to dataframe
            df_with_predictions = df.copy()
            df_with_predictions['predictions'] = predictions
            
            # Generate unique session ID for this prediction
            session_id = str(uuid.uuid4())
            session['current_session_id'] = session_id
            session['filename'] = file.filename
            
            # Save results to file for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"predictions_{timestamp}_{session_id[:8]}.csv"
            results_filepath = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
            df_with_predictions.to_csv(results_filepath, index=False)
            session['results_file'] = results_filepath
            
            print("Predictions completed successfully")
            
            # Convert dataframe to HTML for display
            df_html = df_with_predictions.head(10).to_html(classes='table table-striped table-bordered', 
                                                         table_id='results-table',
                                                         escape=False)
            
            # Prepare summary statistics
            summary_stats = {
                'total_records': len(df_with_predictions),
                'prediction_mean': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'prediction_min': np.min(predictions),
                'prediction_max': np.max(predictions)
            }
            
            return render_template('results.html', 
                                 df_html=df_html, 
                                 summary_stats=summary_stats,
                                 filename=file.filename,
                                 total_records=len(df_with_predictions))
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('predict_file'))

## Route for individual data point prediction (optional - for testing single records)
@app.route('/predict_single', methods=['GET', 'POST'])
def predict_single():
    if request.method == 'GET':
        return render_template('single_predict.html')
    else:
        try:
            # Expected feature names (10 fields from UI)
            expected_input_fields = [
                'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'Neighborhood', 'YrSold',
                'OverallCond', 'LotArea', 'YearBuilt', 'GarageArea', 'YearRemodAdd'
            ]

            # Read training columns to ensure full schema for preprocessor
            train_path = os.path.join('artifacts', 'train.csv')
            train_df = pd.read_csv(train_path)
            feature_cols = train_df.drop(columns=[c for c in ['Id', 'SalePrice'] if c in train_df.columns], errors='ignore').columns.tolist()

            # Initialize a single-row dict with all expected columns as NaN
            row_data = {col: [np.nan] for col in feature_cols}

            # Fill provided values from form
            form = request.form

            def _to_number(value: str):
                if value is None or value == '':
                    return np.nan
                try:
                    # Prefer int when possible
                    iv = int(value)
                    return iv
                except Exception:
                    try:
                        return float(value)
                    except Exception:
                        return value

            for key in expected_input_fields:
                if key in row_data:
                    val = form.get(key, '')
                    # Neighborhood is categorical; keep raw string. Others numeric -> convert
                    if key == 'Neighborhood':
                        row_data[key] = [val]
                    else:
                        row_data[key] = [_to_number(val)]

            # Build dataframe with full schema
            pred_df = pd.DataFrame(row_data)

            # Predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Prepare input data echo-back
            input_summary = {k: form.get(k, '') for k in expected_input_fields}

            return render_template('single_predict.html', 
                                 results=float(results[0]),
                                 input_data=input_summary)
            
        except Exception as e:
            print(f"Error during single prediction: {str(e)}")
            flash(f'Error making prediction: {str(e)}')
            return redirect(url_for('predict_single'))

## Route to download results
@app.route('/download_results')
def download_results():
    try:
        # Check if user has a current session with results
        if 'results_file' not in session:
            flash('No results available for download. Please upload a file first.')
            return redirect(url_for('predict_file'))
        
        results_filepath = session['results_file']
        
        # Check if file exists
        if not os.path.exists(results_filepath):
            flash('Results file not found. Please generate predictions again.')
            return redirect(url_for('predict_file'))
        
        # Get original filename for download
        original_filename = session.get('filename', 'unknown.csv')
        download_filename = f"predictions_{original_filename.split('.')[0]}.csv"
        
        return send_file(results_filepath, 
                        as_attachment=True, 
                        download_name=download_filename,
                        mimetype='text/csv')
        
    except Exception as e:
        print(f"Error downloading results: {str(e)}")
        flash(f'Error downloading results: {str(e)}')
        return redirect(url_for('index'))

## Route for API endpoint (for programmatic access)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        
        if not allowed_file(file.filename):
            return {'error': 'Invalid file type'}, 400
        
        # Process file
        file_handler = FileUploadData(file)
        df = file_handler.process_uploaded_file()
        
        # Get predictions
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(df)
        
        # Return JSON response
        return {
            'success': True,
            'total_records': len(predictions),
            'predictions': predictions.tolist(),
            'summary': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

## Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('predict_file'))

@app.errorhandler(500)
def internal_error(e):
    flash('An internal error occurred. Please try again.')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
