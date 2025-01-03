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
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap='viridis', xticks_rotation=45)
    plt.title(title)
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

        # Aggregate metrics for the current model
        metrics = ['precision', 'recall', 'f1-score']
        avg_results = {
            metric: {
                cls: np.mean([fold[cls][metric] for fold in results if cls in fold]) 
                for cls in map(str, column_mapping.keys())
            }
            for metric in metrics
        }
        overall_accuracy = np.mean([fold['accuracy'] for fold in results])

        # Create a DataFrame for metrics and rename columns
        metrics_df = pd.DataFrame(avg_results).T
        metrics_df.rename(columns=column_mapping, inplace=True)
        metrics_df["Overall Accuracy"] = overall_accuracy
        metrics_results[model.__class__.__name__] = metrics_df

        # Aggregate confusion matrices
        aggregated_conf_matrix = np.sum(conf_matrices, axis=0)
        conf_matrices_results[model.__class__.__name__] = aggregated_conf_matrix

    return metrics_results, conf_matrices_results
