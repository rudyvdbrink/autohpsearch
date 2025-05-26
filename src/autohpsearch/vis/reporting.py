import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Union, Optional


def feature_plot(feature: Union[np.ndarray, pd.Series], 
                 title: Optional[str] = None,
                 max_categories: int = 10,
                 figsize: tuple = (8, 6),
                 color: str = '#4472C4',
                 ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a single feature as a histogram for numerical data or a bar plot for categorical data.
    
    Parameters
    ----------
    feature : array-like or pandas Series
        The feature data to plot
    title : str, optional
        Title for the plot. If None and feature is a pandas Series, uses the Series name.
    max_categories : int, default=10
        Maximum number of categories to display in a bar plot before grouping the rest as 'Other'
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches
    color : str, default='#4472C4'
        Color to use for the plot
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If not provided, creates a new figure and axes.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    # Convert numpy array to pandas Series if needed
    if isinstance(feature, np.ndarray):
        feature = pd.Series(feature)
    
    # Set title if not provided
    if title is None and feature.name is not None:
        title = str(feature.name)
    elif title is None:
        title = "Feature"
        
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Determine if the feature is categorical or numerical
    is_categorical = False
    
    # Check if data type is categorical
    if isinstance(feature.dtype, pd.CategoricalDtype):
        is_categorical = True
    # Check if data type is object or string
    elif pd.api.types.is_object_dtype(feature) or pd.api.types.is_string_dtype(feature):
        is_categorical = True
    # Check if integer with few unique values
    elif pd.api.types.is_numeric_dtype(feature):
        n_unique = feature.nunique()
        n_samples = len(feature)
        # Consider it categorical if there are few unique values relative to sample size
        if n_unique < min(max_categories * 2, n_samples * 0.05):
            is_categorical = True
    
    # Plot based on the data type
    if is_categorical:
        # Get value counts and handle NaN values
        value_counts = feature.value_counts().reset_index()
        value_counts.columns = ['category', 'count']
        
        # Handle too many categories
        if len(value_counts) > max_categories:
            # Keep top categories and group the rest as 'Other'
            top_categories = value_counts.iloc[:max_categories-1]
            other_count = value_counts.iloc[max_categories-1:]['count'].sum()
            
            # Create a new row for 'Other'
            other_row = pd.DataFrame({'category': ['Other'], 'count': [other_count]})
            
            # Combine top categories with 'Other'
            value_counts = pd.concat([top_categories, other_row], ignore_index=True)
            
        # Sort by count in descending order
        value_counts = value_counts.sort_values('count', ascending=False)
        
        # Plot the bar chart
        sns.barplot(x='category', y='count', data=value_counts, ax=ax, color=color)
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        # Improve appearance
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        
        # If there are long category names, adjust figure size
        if value_counts['category'].astype(str).str.len().max() > 10:
            plt.tight_layout()
    else:
        # For numerical data, plot a histogram
        sns.histplot(feature.dropna(), kde=True, ax=ax, color=color)
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')
    
    # Set title
    ax.set_title(title)
    
    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Return the axes
    return ax


def plot_feature_grid(X: Union[np.ndarray, pd.DataFrame], 
                      cols: int = 4,
                      max_categories: int = 10,
                      figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Plot all features in X in a grid layout with fixed number of columns.
    
    Parameters
    ----------
    X : array-like or pandas DataFrame
        The feature matrix to plot
    cols : int, default=4
        Number of columns in the grid (P)
    max_categories : int, default=10
        Maximum number of categories to display in categorical plots
    figsize : tuple, optional
        Figure size as (width, height) in inches. If None, calculated based on grid size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all the plots
    """
    # Convert numpy array to pandas DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    
    # Get the number of features
    n_features = X.shape[1]
    
    # Calculate the number of rows needed in the grid - REPLACED MATH WITH NUMPY
    rows = int(np.ceil(n_features / cols))
    
    # Calculate figsize if not provided
    if figsize is None:
        figsize = (cols * 4, rows * 3)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(rows, cols, figure=fig)
    
    # Use consistent colors across all plots
    colors = plt.cm.tab10.colors
    
    # Create a plot for each feature
    for i, col_name in enumerate(X.columns):
        # Get row and column for the current plot
        row_idx = i // cols
        col_idx = i % cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        # Get the color for this plot
        color = colors[i % len(colors)]
        
        # Plot the feature
        feature_plot(X[col_name], title=col_name, max_categories=max_categories, 
                     ax=ax, color=color)
    
    # Adjust layout
    plt.tight_layout()
    
    # Return the figure
    return fig