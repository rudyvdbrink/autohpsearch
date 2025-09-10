# %% libraries

from sklearn.metrics import (confusion_matrix, 
                             roc_curve, 
                             auc, 
                             classification_report, 
                             balanced_accuracy_score,
                             root_mean_squared_error,
                             )

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %% plots for classification evaluation

def plot_confusion_matrix(y_true, y_score, labels):
    """
    Plot a confusion matrix.
    
    Parameters:
    y_true (array-like): The true labels.
    y_score (array-like): The predicted labels.
    labels (array-like): The class labels or dict of label to number mapping.
    """
    
    # if the labels are a dictionary, assume this is a mapping from labels to integers
    if isinstance(labels, dict):

        if not isinstance(y_true, np.ndarray):
            y_true  = pd.Series(y_true).map(labels['to_numeric'])
        
        if not isinstance(y_score, np.ndarray):
            y_score = pd.Series(y_score).map(labels['to_numeric'])

        labels = list(labels['to_numeric'].keys())        

    #make sure the arrays are numpy arrays and floats
    y_true = np.array(y_true).astype(float)
    y_score = np.array(y_score).astype(float)   

    print(classification_report(y_score,y_true,zero_division=0))
    print('Accuracy = ' + str(np.mean(y_score==y_true)))
    print('Balanced accuracy = ' + str(balanced_accuracy_score(y_true,y_score)))

    #create subplots
    fig, ax = plt.subplots(figsize=(12, 5))

    #make the plot square
    ax.set_aspect('equal', 'box')

    #plot the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_score, normalize='true')
    sns.heatmap(conf_matrix, annot=True, cmap="inferno", vmin=0, vmax=1, 
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Balanced accuracy = ' + str(np.round(balanced_accuracy_score(y_true,y_score)*100)) + '%')

    return fig

def plot_ROC_curve(y_true, y_proba, labels):
    """
    Plot the ROC curve.
    
    Parameters:
    y_true (array-like): The true labels.
    y_proba (array-like): The predicted probability for each of the labels.
    """
    #create subplots
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the ROC curves for each class
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, np.array(y_proba)[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    else:
        for i, label in enumerate(labels):      
            fpr, tpr, _ = roc_curve( (np.array(y_true)==i).astype(int), np.array(y_proba)[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    ax.set_aspect('equal', 'box')
    ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')  #diagonal line
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc='lower right')
    # plt.show()

    return fig

# %% plots for regression evaluation

def regression_prediction_plot(y_true, y_score):
    """
    Evaluate the model by plotting scatter plots of true vs predicted values
    with a least squares regression line and RMSE.
    
    Parameters:
    y_true (array-like): True values for the test set.
    y_score (array-like): Predicted values for the test set.
    """

    #make sure the arrays are numpy arrays and floats
    y_true = np.array(y_true).astype(float)
    y_score = np.array(y_score).astype(float)

    #remove singleton dimensions
    y_true = np.squeeze(y_true)
    y_score = np.squeeze(y_score)

    #make the plot
    fig = plt.figure(figsize=(6, 5))
    rmse_test  = root_mean_squared_error(y_true,  y_score)
    r          = np.corrcoef(y_true,y_score)[0,1]
    r2         = r**2
    print("Test Set Correlation Coefficient:", r)
    print("Test Set R^2 Score:", r2)
    print("Test Set Root Mean Squared Error:", rmse_test)
      
    # Scatter plot for the test set
    plt.scatter(y_true, y_score, alpha=0.5, label='Data')
    
    # Regression line for test set
    m_test, b_test = np.polyfit(y_true, y_score, 1)
    plt.plot(y_true, m_test * y_true + b_test, color='blue', label=f'Fit: y={m_test:.2f}x + {b_test:.2f}')
    
    # Perfect prediction line for reference
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'RMSE: {rmse_test:.2f}')
    plt.legend()    
    plt.tight_layout()
    # plt.show()
    # plt.close()

    return fig

def regression_residual_plot(y_true, y_score):
    """
    Plots the residual plot for regression evaluation.

    Parameters:
    y_true (array-like): The true values of the target variable.
    y_score (array-like): The predicted values of the target variable.
    """

    #make sure the arrays are numpy arrays and floats
    y_true = np.array(y_true).astype(float)
    y_score = np.array(y_score).astype(float)

    # Calculate residuals
    residuals = y_true - y_score

    # Plot residuals
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(y_score, residuals, alpha=0.75)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (days)')
    plt.ylabel('Residuals (days)')
    plt.title('Residual Plot')
    # plt.show()

    return fig

# %% reporting on parameter tuning

def bar_plot_results_df(df, column_name):
    """
    Plots a bar plot for a given column in the dataframe, sorted in descending order.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): The column to plot.
    
    Returns:
        ax: The axis handle of the plot.
    """
    # Sort the dataframe by the specified column in descending order
    sorted_df = df.sort_values(by=column_name, ascending=False)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sorted_df.index, sorted_df[column_name], color='skyblue', edgecolor='none')
    
    # Customize the x-axis labels: replace underscores with spaces and remove _clf/_reg extensions
    formatted_labels = [label.replace('_clf', '').replace('_reg', '').replace('_', ' ').replace('logisticression', 'logistic regression') for label in sorted_df.index]
    ax.set_xticks(np.arange(len(formatted_labels)))
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
    
    # Customize the y-axis label: replace underscores with spaces
    ax.set_ylabel(column_name.replace('_', ' '))
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Return the axis handle
    return ax
# %%
