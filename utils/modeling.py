import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix



def plot_confusion_matrix(conf_matrix, class_labels, title="Aggregated Confusion Matrix"):
    """
    Plots the aggregated confusion matrix from a list of confusion matrices.

    Args:
        conf_matrices (list): List of confusion matrices (one per fold).
        class_labels (list or array): List of class labels for the confusion matrix.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap='viridis', xticks_rotation=45, ax=ax, colorbar=False)
    
    ax.tick_params(axis='x', labelsize=6, length=0, width=0)
    ax.tick_params(axis='y', labelsize=6, length=0, width=0)
    
    ax.set_xlabel("Predicted Labels", fontsize=6)
    ax.set_ylabel("True Labels", fontsize=6)
    
    for text in disp.text_.ravel():
        text.set_fontsize(8)
    
    ax.grid(False)
    ax.set_title(title, fontsize=8)
    plt.show()



def evaluate_models(resampled_data, models):
    """
    Evaluates multiple models on resampled data and returns metrics and confusion matrices.

    Parameters:
        resampled_data (dict): Dictionary where each key is a fold index and values are tuples (X_res, y_res, X_val, y_val).
        models (list): List of initialized model instances to evaluate.

    Returns:
        metrics_results (dict): Dictionary where keys are model names and values are DataFrames of metrics.
        conf_matrices_results (dict): Dictionary where keys are model names and values are aggregated confusion matrices.
    """
    column_mapping = {
    "0.0": 'clear',
    "1.0": 'donbot',
    "2.0": 'fast_flux',
    "3.0": 'neris',
    "4.0": 'qvod',
    "5.0": 'rbot'
    }
    metrics_results = {}
    conf_matrices_results = {}

    for model in models:
        print(f"\nEvaluating model: {model.__class__.__name__}")
        results = []
        conf_matrices = []

        # Loop through resampled data
        for _, (X_res, y_res, X_val, y_val) in resampled_data.items():
            # Train model
            model.fit(X_res, y_res)

            # Predict and evaluate
            y_pred = model.predict(X_val)
            results.append(classification_report(y_val, y_pred, output_dict=True))
            conf_matrices.append(confusion_matrix(y_val, y_pred))
        
        metrics = ['precision', 'recall', 'f1-score']
        avg_results = {
            metric: {
                cls: np.nanmean(
                    [fold.get(cls, {}).get(metric, np.nan) for fold in results]
                )
                for cls in [str(key) for key in column_mapping.keys()]
            }
            for metric in metrics
        }

        # Include overall accuracy
        overall_accuracy = np.mean([fold.get('accuracy', np.nan) for fold in results])

        # Format the metrics as rows with model name prefix
        metrics_with_model_name = {
            f"{model.__class__.__name__}_{metric}": [
                avg_results[metric].get(cls, np.nan) for cls in column_mapping.keys()
            ] + [overall_accuracy]
            for metric in metrics
        }

        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(
            metrics_with_model_name, orient='index', 
            columns=list(column_mapping.values()) + ["Overall Accuracy"]
        )

        # Save the DataFrame for the current model
        metrics_results[model.__class__.__name__] = metrics_df

        # # Aggregate confusion matrices
        aggregated_conf_matrix = np.sum(conf_matrices, axis=0)
        conf_matrices_results[model.__class__.__name__] = aggregated_conf_matrix

    return metrics_results, conf_matrices_results
