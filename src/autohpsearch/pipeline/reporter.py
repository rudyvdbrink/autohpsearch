
# %% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import re

from autohpsearch.vis.reporting import (plot_feature_grid,
                                        plot_feature_correlation, 
                                        plot_nans, 
                                        feature_plot)

# %% the class

class DataReporter:
    """
    A class for generating comprehensive data reports in markdown format.
    """
    
    def __init__(self, report_directory: str = "reports"):
        """
        Initialize the DataReporter.
        
        Parameters
        ----------
        report_directory : str, optional (default="reports")
            Directory where reports will be saved
        """
        self.report_directory = report_directory
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.report_directory, exist_ok=True)
    
    def _generate_descriptive_stats(self, X, y=None):
        """
        Generate descriptive statistics for the dataset.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        y : array-like, optional
            Target variable
            
        Returns
        -------
        dict
            Dictionary containing descriptive statistics
        """
        if hasattr(X, 'columns'):
            data = X
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            data = pd.DataFrame(X, columns=feature_names)
        
        stats = {
            'n_samples': len(data),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'missing_values': {},
            'data_types': {},
            'numeric_features': [],
            'categorical_features': []
        }
        
        # Analyze each feature
        for feature in feature_names:
            # Missing values
            missing_count = data[feature].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            stats['missing_values'][feature] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Data types
            dtype = str(data[feature].dtype)
            stats['data_types'][feature] = dtype
            
            # Categorize as numeric or categorical
            if pd.api.types.is_numeric_dtype(data[feature]):
                stats['numeric_features'].append(feature)
            else:
                stats['categorical_features'].append(feature)
        
        # Target variable statistics
        if y is not None:
            if hasattr(y, 'values'):
                y_values = y.values
            else:
                y_values = np.array(y)
            
            stats['target'] = {
                'n_samples': len(y_values),
                'n_unique': len(np.unique(y_values)),
                'missing_count': np.isnan(y_values).sum() if np.issubdtype(y_values.dtype, np.number) else 0,
                'dtype': str(y_values.dtype)
            }
        
        return stats
    
    def _detect_outliers(self, X, method='zscore', threshold=3.0):
        """
        Detect outliers in the dataset.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        method : str, optional (default='zscore')
            Method for outlier detection
        threshold : float, optional (default=3.0)
            Threshold for outlier detection
            
        Returns
        -------
        dict
            Dictionary containing outlier information
        """
        if hasattr(X, 'select_dtypes'):
            numeric_data = X.select_dtypes(include=[np.number])
        else:
            numeric_data = pd.DataFrame(X)
        
        if len(numeric_data.columns) == 0:
            return {'total_outliers': 0, 'outlier_percentage': 0.0, 'method': method}
        
        outlier_mask = np.zeros(len(numeric_data), dtype=bool)
        
        if method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(numeric_data.fillna(numeric_data.mean())))
            outlier_mask = np.any(z_scores > threshold, axis=1)
        elif method == 'iqr':
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            outlier_mask = np.any(
                (numeric_data < lower_bound) | (numeric_data > upper_bound), axis=1
            )
        
        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_count / len(numeric_data)) * 100
        
        return {
            'total_outliers': outlier_count,
            'outlier_percentage': outlier_percentage,
            'method': method,
            'threshold': threshold
        }
    
    def _save_plot(self, fig_or_ax, filename, dpi=150):
        """
        Save a matplotlib figure or axes to file.
        
        Parameters
        ----------
        fig_or_ax : matplotlib.figure.Figure or matplotlib.axes.Axes
            Figure or axes to save
        filename : str
            Filename for the saved plot
        dpi : int, optional (default=150)
            DPI for saved image
            
        Returns
        -------
        str
            Relative path to saved plot
        """
        # Create plots subdirectory
        plots_dir = os.path.join(self.report_directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Full path for the plot
        plot_path = os.path.join(plots_dir, f"{filename}.png")
        
        # Get the figure
        if hasattr(fig_or_ax, 'figure'):
            # It's an axes
            fig = fig_or_ax.figure
        else:
            # It's a figure
            fig = fig_or_ax
        
        # Save the plot
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Return relative path for markdown
        return f"plots/{filename}.png"
    
    def generate_report(self, 
                       X_train, 
                       y_train, 
                       pipeline=None,
                       X_train_processed=None,
                       y_train_processed=None,
                       version=None):
        """
        Generate a comprehensive data report in markdown format.
        
        Parameters
        ----------
        X_train : array-like or DataFrame
            Training features (before preprocessing)
        y_train : array-like
            Training target (before preprocessing)
        pipeline : AutoMLPipeline, optional
            Fitted pipeline object
        X_train_processed : array-like or DataFrame, optional
            Training features after preprocessing
        y_train_processed : array-like, optional
            Training target after preprocessing
        version : int, optional
            Version number for the report
            
        Returns
        -------
        str
            Path to the generated report file
        """
        # Determine version and filename
        if version is None:
            version = self._get_next_version()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data_report_v{version}_{timestamp}.md"
        report_path = os.path.join(self.report_directory, report_filename)
        
        # Generate statistics
        pre_stats = self._generate_descriptive_stats(X_train, y_train)
        pre_outliers = self._detect_outliers(X_train)
        
        # Start building the report
        report_lines = []
        report_lines.append(f"# Data Analysis Report v{version}")
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Pre-processing section
        report_lines.append("## Pre-Processing Analysis")
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("### Dataset Overview")
        report_lines.append(f"- **Number of samples**: {pre_stats['n_samples']:,}")
        report_lines.append(f"- **Number of features**: {pre_stats['n_features']:,}")
        report_lines.append(f"- **Numeric features**: {len(pre_stats['numeric_features'])}")
        report_lines.append(f"- **Categorical features**: {len(pre_stats['categorical_features'])}")
        report_lines.append("")
        
        # Target variable info
        if 'target' in pre_stats:
            report_lines.append("### Target Variable")
            target_info = pre_stats['target']
            report_lines.append(f"- **Data type**: {target_info['dtype']}")
            report_lines.append(f"- **Unique values**: {target_info['n_unique']}")
            if target_info['missing_count'] > 0:
                report_lines.append(f"- **Missing values**: {target_info['missing_count']}")
            report_lines.append("")
        
        # Missing values analysis
        total_missing = sum(info['count'] for info in pre_stats['missing_values'].values())
        if total_missing > 0:
            report_lines.append("### Missing Values")
            report_lines.append(f"- **Total missing values**: {total_missing:,}")
            report_lines.append("- **Missing values by feature**:")
            for feature, info in pre_stats['missing_values'].items():
                if info['count'] > 0:
                    report_lines.append(f"  - {feature}: {info['count']:,} ({info['percentage']:.2f}%)")
            report_lines.append("")
        
        # Outliers analysis
        if pre_outliers['total_outliers'] > 0:
            report_lines.append("### Outliers Analysis")
            report_lines.append(f"- **Detection method**: {pre_outliers['method']}")
            report_lines.append(f"- **Threshold**: {pre_outliers['threshold']}")
            report_lines.append(f"- **Total outliers detected**: {pre_outliers['total_outliers']:,}")
            report_lines.append(f"- **Outlier percentage**: {pre_outliers['outlier_percentage']:.2f}%")
            report_lines.append("")
        
        # Generate and save plots for pre-processing
        try:
            # Feature distributions
            fig_features = plot_feature_grid(X_train)
            features_plot_path = self._save_plot(fig_features, f"features_pre_v{version}")
            report_lines.append("### Feature Distributions (Pre-processing)")
            report_lines.append(f"![Feature Distributions]({features_plot_path})")
            report_lines.append("")
            
            # Target distribution
            ax_target = feature_plot(y_train)
            target_plot_path = self._save_plot(ax_target, f"target_pre_v{version}")
            report_lines.append("### Target Distribution (Pre-processing)")
            report_lines.append(f"![Target Distribution]({target_plot_path})")
            report_lines.append("")
            
            # Missing values plot
            if total_missing > 0:
                fig_nans = plot_nans(X_train)
                nans_plot_path = self._save_plot(fig_nans, f"missing_values_v{version}")
                report_lines.append("### Missing Values Visualization")
                report_lines.append(f"![Missing Values]({nans_plot_path})")
                report_lines.append("")
        except Exception as e:
            report_lines.append("*Note: Some plots could not be generated due to data characteristics*")
            report_lines.append("")
        
        # Pipeline settings
        if pipeline is not None:
            report_lines.append("## Pipeline Configuration")
            report_lines.append("")
            
            report_lines.append("### Preprocessing Settings")
            report_lines.append(f"- **Task type**: {pipeline.task_type}")
            report_lines.append(f"- **Remove outliers**: {pipeline.remove_outliers}")
            if pipeline.remove_outliers:
                report_lines.append(f"- **Outlier method**: {pipeline.outlier_method}")
                report_lines.append(f"- **Outlier threshold**: {pipeline.outlier_threshold}")
            report_lines.append(f"- **Numerical imputation**: {pipeline.num_imputation_strategy}")
            report_lines.append(f"- **Categorical imputation**: {pipeline.cat_imputation_strategy}")
            report_lines.append(f"- **Categorical encoding**: {pipeline.cat_encoding_method}")
            if pipeline.cat_encoding_method == 'auto':
                report_lines.append(f"- **Max one-hot cardinality**: {pipeline.max_onehot_cardinality}")
            report_lines.append(f"- **Scaling method**: {pipeline.scaling_method}")
            if pipeline.task_type == 'regression':
                report_lines.append(f"- **Target transformation**: {pipeline.target_transform}")
            report_lines.append("")
            
            # Model settings
            report_lines.append("### Model Training Settings")
            report_lines.append(f"- **Scoring metric**: {pipeline.scoring}")
            report_lines.append(f"- **Search type**: {pipeline.search_type}")
            report_lines.append(f"- **Number of iterations**: {pipeline.n_iter}")
            report_lines.append(f"- **Cross-validation folds**: {pipeline.cv}")
            report_lines.append("")
            
            # Best model results (if available)
            if hasattr(pipeline, 'best_model_') and pipeline.best_model_ is not None:
                report_lines.append("### Best Model Results")
                report_lines.append(f"- **Best model**: {type(pipeline.best_model_).__name__}")
                
                if hasattr(pipeline, 'results_') and pipeline.results_ is not None:
                    results_df = pipeline.results_['results']
                    best_score = results_df['test_score'].iloc[0]
                    report_lines.append(f"- **Test score ({pipeline.scoring})**: {best_score:.4f}")
                
                # Model parameters
                if hasattr(pipeline.best_model_, 'get_params'):
                    params = pipeline.best_model_.get_params()
                    report_lines.append("- **Best hyperparameters**:")
                    for param, value in params.items():
                        report_lines.append(f"  - {param}: {value}")
                report_lines.append("")
        
        # Post-processing section (only if processed data is available)
        if X_train_processed is not None:
            report_lines.append("## Post-Processing Analysis")
            report_lines.append("")
            
            post_stats = self._generate_descriptive_stats(X_train_processed, y_train_processed)
            
            # Dataset overview after processing
            report_lines.append("### Dataset Overview (After Processing)")
            report_lines.append(f"- **Number of samples**: {post_stats['n_samples']:,}")
            report_lines.append(f"- **Number of features**: {post_stats['n_features']:,}")
            
            # Show changes in sample size
            if post_stats['n_samples'] != pre_stats['n_samples']:
                removed_samples = pre_stats['n_samples'] - post_stats['n_samples']
                removed_pct = (removed_samples / pre_stats['n_samples']) * 100
                report_lines.append(f"- **Samples removed**: {removed_samples:,} ({removed_pct:.2f}%)")
            
            # Show changes in feature count
            if post_stats['n_features'] != pre_stats['n_features']:
                feature_change = post_stats['n_features'] - pre_stats['n_features']
                report_lines.append(f"- **Feature count change**: {feature_change:+d}")
            
            report_lines.append("")
            
            try:
                # Feature correlation plot (post-processing)
                fig_corr = plot_feature_correlation(X_train_processed)
                corr_plot_path = self._save_plot(fig_corr, f"correlation_post_v{version}")
                report_lines.append("### Feature Correlations (Post-processing)")
                report_lines.append(f"![Feature Correlations]({corr_plot_path})")
                report_lines.append("")
                
                # Feature distributions (post-processing)
                fig_features_post = plot_feature_grid(X_train_processed)
                features_post_plot_path = self._save_plot(fig_features_post, f"features_post_v{version}")
                report_lines.append("### Feature Distributions (Post-processing)")
                report_lines.append(f"![Feature Distributions Post]({features_post_plot_path})")
                report_lines.append("")
                
                # Target distribution (post-processing, if different)
                if y_train_processed is not None:
                    ax_target_post = feature_plot(y_train_processed)
                    target_post_plot_path = self._save_plot(ax_target_post, f"target_post_v{version}")
                    report_lines.append("### Target Distribution (Post-processing)")
                    report_lines.append(f"![Target Distribution Post]({target_post_plot_path})")
                    report_lines.append("")
            except Exception as e:
                report_lines.append("*Note: Some post-processing plots could not be generated*")
                report_lines.append("")
        
        # Write the report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return report_path
    
    def _get_next_version(self):
        """Get the next version number for reports."""
        if not os.path.exists(self.report_directory):
            return 1
        
        # Look for existing reports to determine next version
        pattern = re.compile(r"data_report_v(\d+)_.*\.md$")
        version = 1
        
        for filename in os.listdir(self.report_directory):
            match = pattern.match(filename)
            if match:
                v = int(match.group(1))
                version = max(version, v + 1)
        
        return version