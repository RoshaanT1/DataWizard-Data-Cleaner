from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import json
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app = Flask(__name__)
CORS(app)

# Global variable to store the uploaded dataframe
global_df = None
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_data(df):
    """Basic data cleaning operations"""
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

def detect_outliers(df):
    outlier_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_stats[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    return outlier_stats

def get_column_stats(df):
    """Generate comprehensive statistics for each column"""
    stats = {}
    
    for column in df.columns:
        column_stats = {
            'missing': int(df[column].isna().sum()),
            'missing_percentage': float(df[column].isna().mean() * 100),
            'dtype': str(df[column].dtype)
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            column_stats['type'] = 'numeric'
            column_stats.update({
                'mean': float(df[column].mean()),
                'median': float(df[column].median()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mode': float(df[column].mode()[0]) if not df[column].mode().empty else None
            })
        else:
            column_stats['type'] = 'categorical'
            column_stats.update({
                'top_values': {str(k): int(v) for k, v in df[column].value_counts().head(5).to_dict().items()},
                'unique_count': int(df[column].nunique()),
                'mode': str(df[column].mode()[0]) if not df[column].mode().empty else None
            })
        
        stats[column] = column_stats
    
    return stats



def generate_missing_values_summary(df):
    missing_data = df.isna().sum().to_dict()
    total_missing = sum(missing_data.values())
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100
    
    return {
        'missing_per_column': missing_data,
        'total_missing': total_missing,
        'missing_percentage': missing_percentage
    }

def generate_correlation_matrix(df):
    """Generate correlation matrix for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {}
    
    try:
        corr_matrix = numeric_df.corr().round(2)
        # Replace NaN values with 0 and convert to dictionary
        return corr_matrix.fillna(0).to_dict()
    except Exception as e:
        print(f"Error generating correlation matrix: {str(e)}")
        return {}
    
def generate_basic_plots(df):
    """Generate some basic plots based on the data types"""
    plots = {}
    
    # For numeric columns, create histograms
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        for column in numeric_columns[:3]:  # Limit to first 3 columns for demo
            plt.figure(figsize=(8, 5))
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots[f'{column}_histogram'] = plot_data
            plt.close()
    
    # Create a correlation heatmap if there are numeric columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plots['correlation_heatmap'] = plot_data
        plt.close()
    
    # For categorical columns, create countplots for the top categories
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        for column in categorical_columns[:2]:  # Limit to first 2 columns for demo
            plt.figure(figsize=(10, 6))
            top_categories = df[column].value_counts().nlargest(10).index
            sns.countplot(y=column, data=df[df[column].isin(top_categories)], order=top_categories)
            plt.title(f'Top Categories in {column}')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots[f'{column}_countplot'] = plot_data
            plt.close()
    
    return plots
def calculate_quality_score(missing_summary, outlier_stats):
    missing_penalty = missing_summary['missing_percentage'] * 2
    outlier_penalty = sum(
        stats['percentage'] for stats in outlier_stats.values()
    ) * 0.5
    return max(0, 100 - missing_penalty - outlier_penalty)
@app.route('/upload', methods=['POST'])
def upload_file():
    global global_df
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read original data
        if filename.endswith('.csv'):
            original_df = pd.read_csv(file_path)
        elif filename.endswith(('.xls', '.xlsx')):
            original_df = pd.read_excel(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Clean the data
        cleaned_df = clean_data(original_df.copy())
        global_df = cleaned_df
        
        # Get statistics for both original and cleaned data
        original_stats = get_column_stats(original_df)
        cleaned_stats = get_column_stats(cleaned_df)
        original_outliers = detect_outliers(original_df)
        cleaned_outliers = detect_outliers(cleaned_df)
        missing_summary = generate_missing_values_summary(original_df)
        
        # Prepare before and after comparison data
        before_cleaning = {
            'numeric_columns': {
                col: {
                    'mean': stats['mean'] if stats['type'] == 'numeric' else None,
                    'median': stats['median'] if stats['type'] == 'numeric' else None,
                    'mode': stats['mode'] if stats['type'] == 'numeric' else None,
                    'outliers': original_outliers.get(col, {'count': 0, 'percentage': 0.0})
                }
                for col, stats in original_stats.items()
            }
        }
        after_cleaning = {
            'numeric_columns': {
                col: {
                    'mean': stats['mean'] if stats['type'] == 'numeric' else None,
                    'median': stats['median'] if stats['type'] == 'numeric' else None,
                    'mode': stats['mode'] if stats['type'] == 'numeric' else None,
                    'outliers': cleaned_outliers.get(col, {'count': 0, 'percentage': 0.0})
                }
                for col, stats in cleaned_stats.items()
            }
        }
        
        response = {
            'message': 'File uploaded successfully',
            'filename': filename,
            'rows': len(original_df),
            'columns': list(original_df.columns),
            'sample_data': cleaned_df.head(5).to_dict('records'),
            'original_stats': original_stats,
            'cleaned_stats': cleaned_stats,
            'original_outliers': original_outliers,
            'cleaned_outliers': cleaned_outliers,
            'missing_values_summary': missing_summary,
            'data_quality': {
                'total_missing': missing_summary['total_missing'],
                'missing_percentage': missing_summary['missing_percentage'],
                'quality_score': calculate_quality_score(missing_summary, original_outliers)
            },
            'before_cleaning': before_cleaning,
            'after_cleaning': after_cleaning
        }
        
        return jsonify(response)
        
@app.route('/save', methods=['POST'])
def save_data():
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'No data to save'}), 400
    
    try:
        # Get format from request
        format = request.json.get('format', 'csv')
        filename = request.json.get('filename', 'cleaned_data')
        
        # Create BytesIO buffer
        buffer = BytesIO()
        
        if format == 'csv':
            global_df.to_csv(buffer, index=False)
            mimetype = 'text/csv'
            extension = '.csv'
        elif format == 'excel':
            global_df.to_excel(buffer, index=False)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            extension = '.xlsx'
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        buffer.seek(0)
        
        # Return the file for download
        return send_file(
            buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"{filename}{extension}"
        )
    
    except Exception as e:
        return jsonify({'error': f'Error saving data: {str(e)}'}), 500
@app.route('/analyze', methods=['GET'])
def analyze_data():
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
    
    # Generate correlation matrix
    correlation_matrix = generate_correlation_matrix(global_df)
    
    # Generate basic plots
    plots = generate_basic_plots(global_df)
    
    # Sample insights based on correlations
    insights = []
    if correlation_matrix and isinstance(correlation_matrix, dict):
        for col1 in correlation_matrix:
            # Skip if the column doesn't exist in the matrix
            if col1 not in correlation_matrix:
                continue
                
            for col2, corr_value in correlation_matrix[col1].items():
                if col1 != col2 and abs(corr_value) > 0.7:
                    insights.append({
                        'type': 'correlation',
                        'description': f'Strong {"positive" if corr_value > 0 else "negative"} correlation ({corr_value}) detected between {col1} and {col2}',
                        'strength': abs(corr_value)
                    })
    
    response = {
        'correlation_matrix': correlation_matrix if correlation_matrix else {},
        'insights': insights,
        'plots': plots
    }
    
    return jsonify(response)

from llama_cpp import Llama
llm = Llama(
    model_path="deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
    n_ctx=1024,  # Reduced context for memory savings
    n_threads=4,  # Use 4 of your 8 CPU threads
    n_batch=256,  # Smaller batch size = less RAM
)

@app.route('/query', methods=['POST'])
def query_data():
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
    
    query = request.json.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    # First check for predefined simple queries (fast response)
    if 'how many rows' in query.lower():
        return jsonify({
            "answer": f"The dataset contains {len(global_df)} rows.",
            "type": "predefined"
        })
        
    elif 'how many columns' in query.lower():
        return jsonify({
            "answer": f"The dataset contains {len(global_df.columns)} columns: {', '.join(global_df.columns)}.",
            "type": "predefined"
        })
        
    elif 'missing values' in query.lower():
        missing = global_df.isna().sum().to_dict()
        return jsonify({
            "answer": f"Missing values by column:\n{json.dumps(missing, indent=2)}",
            "type": "predefined"
        })
        
    elif 'summary' in query.lower():
        numeric_summary = global_df.describe().to_dict()
        return jsonify({
            "answer": f"Numeric columns summary:\n{json.dumps(numeric_summary, indent=2)}",
            "type": "predefined"
        })

    # For all other queries, use the local DeepSeek model
    try:
        # Generate context about the data
        data_context = {
            "shape": f"{len(global_df)} rows, {len(global_df.columns)} columns",
            "columns": list(global_df.columns),
            "numeric_columns": global_df.select_dtypes(include='number').columns.tolist(),
            "missing_values": global_df.isna().sum().to_dict()
        }

        # Create optimized prompt
        prompt = f"""<|im_start|>system
You are a data analysis assistant working with a pandas DataFrame.
Dataset Info: {json.dumps(data_context, indent=2)}

Guidelines:
1. Be concise (1-3 sentences)
2. Format tables as markdown
3. Highlight key numbers in **bold**
4. If suggesting analysis, recommend only 1-2 simple ones<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""

        # Get LLM response (with reduced tokens for 8GB RAM)
        response = llm.create_completion(
            prompt,
            max_tokens=200,
            temperature=0.3,
            stop=["<|im_end|>"]
        )

        return jsonify({
            "answer": response["choices"][0]["text"].strip(),
            "type": "llm",
            "context": data_context
        })

    except Exception as e:
        return jsonify({
            "error": f"AI processing failed: {str(e)}",
            "fallback_answer": f"I couldn't process that. Basic info: {len(global_df)} rows, {len(global_df.columns)} columns."
        }), 500

@app.route('/pivot', methods=['POST'])
def create_pivot():
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
    
    try:
        pivot_config = request.json.get('pivot_config', {})
        
        # Extract pivot parameters
        values = pivot_config.get('values', [])
        index = pivot_config.get('index', [])
        columns = pivot_config.get('columns', [])
        aggfunc = pivot_config.get('aggfunc', 'mean')
        
        # Create pivot table
        pivot_df = pd.pivot_table(
            global_df,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=0
        ).reset_index()
        
        # Convert pivot table to records format for JSON response
        pivot_data = {
            'data': pivot_df.to_dict('records'),
            'columns': [{'name': col, 'id': col} for col in pivot_df.columns]
        }
        
        return jsonify(pivot_data)
    
    except Exception as e:
        return jsonify({'error': f'Error creating pivot table: {str(e)}'}), 500

@app.route('/filter', methods=['POST'])
def filter_data():
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
    
    try:
        filters = request.json.get('filters', {})
        cleaning_ops = request.json.get('cleaning', {})
        
        filtered_df = global_df.copy()
        outliers_info = []
        
        # Apply filters (existing functionality)
        for column, condition in filters.items():
            if column in filtered_df.columns:
                if condition['type'] == 'range':
                    if pd.api.types.is_numeric_dtype(filtered_df[column]):
                        min_val = condition.get('min')
                        max_val = condition.get('max')
                        if min_val is not None:
                            filtered_df = filtered_df[filtered_df[column] >= min_val]
                        if max_val is not None:
                            filtered_df = filtered_df[filtered_df[column] <= max_val]
                elif condition['type'] == 'category':
                    selected_values = condition.get('values', [])
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        
        # Apply cleaning operations
        if 'missing_values' in cleaning_ops:
            mv_ops = cleaning_ops['missing_values']
            columns = mv_ops.get('columns', [])
            method = mv_ops.get('method', 'mean')
            
            for column in columns:
                if column in filtered_df.columns:
                    if method == 'drop':
                        filtered_df = filtered_df.dropna(subset=[column])
                    elif method == 'mean' and pd.api.types.is_numeric_dtype(filtered_df[column]):
                        filtered_df[column] = filtered_df[column].fillna(filtered_df[column].mean())
                    elif method == 'median' and pd.api.types.is_numeric_dtype(filtered_df[column]):
                        filtered_df[column] = filtered_df[column].fillna(filtered_df[column].median())
                    elif method == 'mode':
                        filtered_df[column] = filtered_df[column].fillna(
                            filtered_df[column].mode()[0] if not filtered_df[column].mode().empty else "Unknown"
                        )
                    elif method == 'zero' and pd.api.types.is_numeric_dtype(filtered_df[column]):
                        filtered_df[column] = filtered_df[column].fillna(0)
        
        if 'outliers' in cleaning_ops:
            outlier_ops = cleaning_ops['outliers']
            columns = outlier_ops.get('columns', [])
            method = outlier_ops.get('method', 'iqr')
            threshold = float(outlier_ops.get('threshold', 1.5))
            treatment = outlier_ops.get('treatment', 'remove')
            
            for column in columns:
                if column in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[column]):
                    q1 = filtered_df[column].quantile(0.25)
                    q3 = filtered_df[column].quantile(0.75)
                    iqr = q3 - q1
                    
                    if method == 'iqr':
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                    else:  # zscore
                        mean = filtered_df[column].mean()
                        std = filtered_df[column].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                    
                    outlier_mask = (filtered_df[column] < lower_bound) | (filtered_df[column] > upper_bound)
                    outliers = filtered_df[outlier_mask][[column]].copy()
                    
                    # Store detailed outlier information
                    for idx, row in outliers.iterrows():
                        original_value = row[column]
                        new_value = original_value  # Default to original value
                        action = ""
                        
                        if treatment == 'remove':
                            action = f"Removed outlier {original_value}"
                        elif treatment == 'mean':
                            new_value = filtered_df[column].mean()
                            action = f"Replaced outlier {original_value} with mean {new_value:.2f}"
                        elif treatment == 'median':
                            new_value = filtered_df[column].median()
                            action = f"Replaced outlier {original_value} with median {new_value:.2f}"
                        elif treatment == 'mode':
                            mode_val = filtered_df[column].mode()[0] if not filtered_df[column].mode().empty else 0
                            new_value = mode_val
                            action = f"Replaced outlier {original_value} with mode {new_value:.2f}"
                        elif treatment == 'cap':
                            new_value = lower_bound if original_value < lower_bound else upper_bound
                            action = f"Capped outlier {original_value} to {new_value:.2f}"
                        
                        outliers_info.append({
                            'column': column,
                            'row_index': int(idx),
                            'original_value': float(original_value),
                            'new_value': float(new_value) if new_value is not None else None,
                            'treatment': action,
                            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                        })
                    
                    # Apply treatment to the dataset
                    if treatment == 'remove':
                        filtered_df = filtered_df[~outlier_mask]
                    elif treatment == 'mean':
                        filtered_df.loc[outlier_mask, column] = filtered_df[column].mean()
                    elif treatment == 'median':
                        filtered_df.loc[outlier_mask, column] = filtered_df[column].median()
                    elif treatment == 'mode':
                        mode_val = filtered_df[column].mode()[0] if not filtered_df[column].mode().empty else 0
                        filtered_df.loc[outlier_mask, column] = mode_val
                    elif treatment == 'cap':
                        filtered_df.loc[filtered_df[column] < lower_bound, column] = lower_bound
                        filtered_df.loc[filtered_df[column] > upper_bound, column] = upper_bound
        
        # Update global_df with cleaned data
        global_df = filtered_df
        
        # Get updated statistics
        cleaned_stats = get_column_stats(filtered_df)
        cleaned_outliers = detect_outliers(filtered_df)
        
        # Prepare after cleaning stats
        after_cleaning = {
            'numeric_columns': {
                col: {
                    'mean': stats['mean'] if stats['type'] == 'numeric' else None,
                    'median': stats['median'] if stats['type'] == 'numeric' else None,
                    'mode': stats['mode'] if stats['type'] == 'numeric' else None,
                    'outliers': cleaned_outliers.get(col, {'count': 0, 'percentage': 0.0})
                }
                for col, stats in cleaned_stats.items()
            }
        }
        
        return jsonify({
            'data': filtered_df.head(1000).to_dict('records'),
            'columns': [{'name': col, 'id': col} for col in filtered_df.columns],
            'cleaning_results': {
                'missing_values': 'Applied' if 'missing_values' in cleaning_ops else None,
                'outliers': outliers_info if outliers_info else None
            },
            'after_cleaning': after_cleaning
        })
    
    except Exception as e:
        return jsonify({'error': f'Error filtering/cleaning data: {str(e)}'}), 500
        
from flask import render_template
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)