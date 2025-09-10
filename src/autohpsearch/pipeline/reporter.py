
# %% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import re

from autohpsearch.utils.context import hush

from autohpsearch.vis.reporting import (plot_feature_grid,
                                        plot_feature_correlation, 
                                        plot_nans, 
                                        target_plot,
                                        plot_design_matrix)

from autohpsearch.vis.evaluation_plots import (regression_prediction_plot,
                                               regression_residual_plot,
                                               plot_confusion_matrix,
                                               plot_ROC_curve,
                                               bar_plot_results_df)


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
    
    def _generate_descriptive_stats(self, X, y=None, feature_names=None):
        """
        Generate descriptive statistics for the dataset.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        y : array-like, optional
            Target variable
        feature_names : list, optional
            Feature names to use (useful for processed data with known feature names)
            
        Returns
        -------
        dict
            Dictionary containing descriptive statistics
        """
        if hasattr(X, 'columns'):
            data = X
            feature_names_to_use = X.columns.tolist()
        else:
            # For numpy arrays, use provided feature names or create generic ones
            if feature_names is not None:
                feature_names_to_use = feature_names
            else:
                feature_names_to_use = [f'Feature_{i}' for i in range(X.shape[1])]
            data = pd.DataFrame(X, columns=feature_names_to_use)
        
        stats = {
            'n_samples': len(data),
            'n_features': len(feature_names_to_use),
            'feature_names': feature_names_to_use,
            'missing_values': {},
            'data_types': {},
            'numeric_features': [],
            'categorical_features': []
        }
        
        # Analyze each feature
        for feature in feature_names_to_use:
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
    
    def _detect_outliers(self, X, y, pipeline=None):
        """
        Detect outliers in the data using the same method as in the pipeline.
        
        Parameters
        ----------
        X : array-like or DataFrame
            The input data to detect outliers in
        pipeline : AutoMLPipeline, optional
            The fitted pipeline that may contain an outlier remover
            
        Returns
        -------
        dict
            Dictionary containing outlier statistics and detection parameters:
            - total_outliers: Number of outliers detected
            - outlier_percentage: Percentage of data points identified as outliers
            - method: Method used for outlier detection
            - threshold: Threshold value used for detection
        """
        # Default values
        n_removed = 0
        outlier_percentage = 0.0
        method = 'none'
        threshold = 0.0
        
        if pipeline.remove_outliers is False:
            return {
            'total_outliers': n_removed,
            'outlier_percentage': outlier_percentage,
            'method': method,
            'threshold': threshold
        }

        # Check if pipeline has an outlier remover
        if pipeline is not None and hasattr(pipeline, 'outlier_remover_') and pipeline.outlier_remover_ is not None:
            
            from autohpsearch.pipeline.cleaning import OutlierRemover

            outlier_remover = OutlierRemover(
                method=pipeline.outlier_method,
                threshold=pipeline.outlier_threshold
            )
            
            # Get N for reference
            y_len = len(y)
            
            # Fit and transform on training data only
            outlier_remover.fit(X)
            X, y = outlier_remover.transform(X, y)            

            n_removed = y_len - len(y)
            outlier_percentage = n_removed/y_len*100
            method=pipeline.outlier_method
            threshold=pipeline.outlier_threshold
            
        return {
            'total_outliers': n_removed,
            'outlier_percentage': outlier_percentage,
            'method': method,
            'threshold': threshold
        }
        
    def _save_plot(self, fig_or_ax, filename, report_subfolder, dpi=150):
        """
        Save a matplotlib figure or axes to file.
        
        Parameters
        ----------
        fig_or_ax : matplotlib.figure.Figure or matplotlib.axes.Axes
            Figure or axes to save
        filename : str
            Filename for the saved plot
        report_subfolder : str
            Subfolder name for this specific report
        dpi : int, optional (default=150)
            DPI for saved image
            
        Returns
        -------
        str
            Relative path to saved plot (for markdown)
        """
        # Create plots subdirectory inside the report subfolder
        plots_dir = os.path.join(self.report_directory, report_subfolder, "plots")
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
        
        # Return relative path for markdown (relative to the report file location)
        return f"plots/{filename}.png"
    
    def generate_report(self, 
                       X_train, 
                       y_train, 
                       pipeline=None,
                       X_train_processed=None,
                       y_train_processed=None,
                       X_test=None,
                       y_test=None,
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
        X_test : array-like or DataFrame
            Testing features 
        y_test : array-like
            Testing target 
        version : int, optional
            Version number for the report
            
        Returns
        -------
        str
            Path to the generated report file
        """
        # Determine version and filename
        # Determine version and filename
        if version is None:
            version = self._get_next_version()
        else:
            version = f"{version:04d}"  # Ensure zero-padding
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename_base = f"data_report_v{version}_{timestamp}"
        report_filename = f"{report_filename_base}.md"
        
        # Create subfolder for this report
        report_subfolder = report_filename_base
        report_subfolder_path = os.path.join(self.report_directory, report_subfolder)
        os.makedirs(report_subfolder_path, exist_ok=True)
        
        # Full path for the report file (inside the subfolder)
        report_path = os.path.join(report_subfolder_path, report_filename)
        
        # Generate statistics
        pre_stats = self._generate_descriptive_stats(X_train, y_train)
        pre_outliers = self._detect_outliers(X_train,y_train,pipeline=pipeline)
        
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
        
        # Feature names section
        if pre_stats['feature_names']:
            report_lines.append("### Original Feature Names")
            if len(pre_stats['feature_names']) <= 20:
                feature_list = ', '.join(f"`{name}`" for name in pre_stats['feature_names'])
                report_lines.append(f"- {feature_list}")
            else:
                report_lines.append(f"- **Total features**: {len(pre_stats['feature_names'])}")
                report_lines.append(f"- **First 10**: {', '.join(f'`{name}`' for name in pre_stats['feature_names'][:10])}")
                report_lines.append(f"- **Last 10**: {', '.join(f'`{name}`' for name in pre_stats['feature_names'][-10:])}")
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
            features_plot_path = self._save_plot(fig_features, f"features_pre_v{version}", report_subfolder)
            report_lines.append("### Feature Distributions (Pre-processing)")
            report_lines.append(f"![Feature Distributions]({features_plot_path})")
            report_lines.append("")
            
            # Target distribution 
            ax_target = target_plot(y_train, title="Target")
            target_plot_path = self._save_plot(ax_target, f"target_pre_v{version}", report_subfolder)
            report_lines.append("### Target Distribution (Pre-processing)")
            report_lines.append(f"![Target Distribution]({target_plot_path})")
            report_lines.append("")
            
            # Missing values plot
            if total_missing > 0:
                fig_nans = plot_nans(X_train)
                nans_plot_path = self._save_plot(fig_nans, f"missing_values_v{version}", report_subfolder)
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
            report_lines.append(f"- **Drop duplicate rows**: {pipeline.drop_duplicate_rows}")
            if pipeline.filter_features:
                report_lines.append(f"- **Feature filtering**: {pipeline.filter_features}")
                report_lines.append(f"- **Feature correlation threshold**: {pipeline.filter_threshold}")
            if pipeline.remove_outliers:
                report_lines.append(f"- **Outlier method**: {pipeline.outlier_method}")
                report_lines.append(f"- **Outlier threshold**: {pipeline.outlier_threshold}")
            report_lines.append(f"- **Numerical imputation**: {pipeline.num_imputation_strategy}")
            report_lines.append(f"- **Categorical imputation**: {pipeline.cat_imputation_strategy}")
            report_lines.append(f"- **Categorical encoding**: {pipeline.cat_encoding_method}")
            if pipeline.cat_encoding_method == 'auto':
                report_lines.append(f"- **Max one-hot cardinality**: {pipeline.max_onehot_cardinality}")
            report_lines.append(f"- **Scaling method**: {pipeline.scaling_method}")
            if pipeline.task_type == 'classification':
                report_lines.append(f"- **SMOTE oversampling**: {pipeline.apply_smote}")
                report_lines.append(f"- **SMOTE kwargs**: {pipeline.smote_kwargs}")
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
        
        # Post-processing section (only if processed data is available)
        if X_train_processed is not None:
            report_lines.append("## Post-Processing Analysis")
            report_lines.append("")
            
            # Get transformed feature names from pipeline if available
            transformed_feature_names = None
            if pipeline is not None and hasattr(pipeline, 'get_feature_names'):
                try:
                    transformed_feature_names = pipeline.get_feature_names('transformed')
                except:
                    transformed_feature_names = None
            
            post_stats = self._generate_descriptive_stats(X_train_processed, y_train_processed, 
                                                         feature_names=transformed_feature_names)
            
            # Dataset overview after processing
            report_lines.append("### Dataset Overview (After Processing)")
            report_lines.append(f"- **Number of samples**: {post_stats['n_samples']:,}")
            report_lines.append(f"- **Number of features**: {post_stats['n_features']:,}")
            
            # Show changes in sample size
            if post_stats['n_samples'] != pre_stats['n_samples']:
                removed_samples = pre_stats['n_samples'] - post_stats['n_samples']
                removed_pct = (removed_samples / pre_stats['n_samples']) * 100
                report_lines.append(f"- **Samples removed**: {removed_samples:,} ({removed_pct:.2f}%)")

                if pipeline.drop_duplicate_rows:
                    num_rows_dropped = pipeline.preprocessor_.num_rows_dropped_
                    print(f"- **Rows dropped due to duplicates**: {num_rows_dropped:,}")

            # Show changes in feature count
            if post_stats['n_features'] != pre_stats['n_features']:
                feature_change = post_stats['n_features'] - pre_stats['n_features']
                report_lines.append(f"- **Feature count change**: {feature_change:+d}")
            
            report_lines.append("")
            
            # Transformed feature names section
            if transformed_feature_names:
                report_lines.append("### Transformed Feature Names")
                if len(transformed_feature_names) <= 20:
                    feature_list = ', '.join(f"`{name}`" for name in transformed_feature_names)
                    report_lines.append(f"- {feature_list}")
                else:
                    report_lines.append(f"- **Total features**: {len(transformed_feature_names)}")
                    report_lines.append(f"- **First 10**: {', '.join(f'`{name}`' for name in transformed_feature_names[:10])}")
                    report_lines.append(f"- **Last 10**: {', '.join(f'`{name}`' for name in transformed_feature_names[-10:])}")
                                
                report_lines.append("")

            if pipeline and pipeline.filter_features:
                report_lines.append("### Feature Filtering")
                if pipeline.columns_to_drop_ is not None:
                    report_lines.append(f"- **Number of features removed**: {len(pipeline.columns_to_drop_)}")
                    report_lines.append(f"- **Removed features**: {', '.join(f'`{col}`' for col in pipeline.columns_to_drop_)}")
                else:
                    report_lines.append("- **No features were removed**")
                report_lines.append("")
            
            try:
                # Create DataFrame with feature names for post-processing plots
                if transformed_feature_names and not hasattr(X_train_processed, 'columns'):
                    X_processed_df = pd.DataFrame(X_train_processed, columns=transformed_feature_names)
                else:
                    X_processed_df = X_train_processed
                
                # Feature correlation plot (post-processing)
                fig_corr = plot_feature_correlation(X_processed_df)
                corr_plot_path = self._save_plot(fig_corr, f"correlation_post_v{version}", report_subfolder)
                report_lines.append("### Feature Correlations (Post-processing)")
                report_lines.append(f"![Feature Correlations]({corr_plot_path})")
                report_lines.append("")
                
                # Feature distributions (post-processing)
                fig_features_post = plot_feature_grid(X_processed_df)
                features_post_plot_path = self._save_plot(fig_features_post, f"features_post_v{version}", report_subfolder)
                report_lines.append("### Feature Distributions (Post-processing)")
                report_lines.append(f"![Feature Distributions Post]({features_post_plot_path})")
                report_lines.append("")

                # Design matrix plot (post-processing)
                design_matrix_figure = plot_design_matrix(X_processed_df)
                design_matrix_path = self._save_plot(design_matrix_figure, f"design_matrix_post_v{version}", report_subfolder)
                report_lines.append("### Design Matrix (Post-processing)")
                report_lines.append(f"![Design Matrix]({design_matrix_path})")
                report_lines.append("")
                
                # Target distribution (post-processing, if different)
                if y_train_processed is not None:
                    ax_target_post = target_plot(y_train_processed, title="Target (Processed)",  labels=pipeline.label_mapping_)
                    target_post_plot_path = self._save_plot(ax_target_post, f"target_post_v{version}", report_subfolder)
                    report_lines.append("### Target Distribution (Post-processing)")
                    report_lines.append(f"![Target Distribution Post]({target_post_plot_path})")
                    report_lines.append("")
            except Exception as e:
                report_lines.append("*Note: Some post-processing plots could not be generated*")
                report_lines.append("")
        
        # Overview of the pipeline results
        if hasattr(pipeline, 'results_') and pipeline.results_ is not None:

            results_df = pipeline.results_['results']
            
            # Convert the DataFrame to a Markdown table
            results_table = results_df.to_markdown(index=True, tablefmt="pipe", floatfmt=".4f")
            
            # Add the table to the report
            report_lines.append("## Model Comparison")
            report_lines.append("")
            report_lines.append("### Model Comparison Table")
            report_lines.append("")
            report_lines.append(results_table)
            report_lines.append("")

            # Plot cross-validation performance
            ax = bar_plot_results_df(pipeline.results_['results'], 'cv_score')
            cv_plot_path = self._save_plot(ax, f"cv_performance_v{version}", report_subfolder)
            report_lines.append("### Cross-Validation Performance")
            report_lines.append(f"![Cross-Validation Performance]({cv_plot_path})")
            report_lines.append("") 

            # Plot timing information
            ax = bar_plot_results_df(pipeline.results_['results'], 'train_time_ms')
            timing_plot_path = self._save_plot(ax, f"timing_v{version}", report_subfolder)
            report_lines.append("### Training Time per Model Variant")
            report_lines.append(f"![Training Time]({timing_plot_path})")
            report_lines.append("")         

        # Best Model Results Section 
        if pipeline is not None and hasattr(pipeline, 'best_model_') and pipeline.best_model_ is not None:
            report_lines.append("## Best Model Results")
            report_lines.append("")
            
            report_lines.append("### Model Information")
            report_lines.append(f"- **Best model**: {type(pipeline.best_model_).__name__}")
            
            if hasattr(pipeline, 'results_') and pipeline.results_ is not None:
                results_df = pipeline.results_['results']
                best_score = results_df['test_score'].iloc[0]
                report_lines.append(f"- **Test score ({pipeline.scoring})**: {best_score:.4f}")
                
                # Add CV score if available
                if 'cv_score_mean' in results_df.columns:
                    cv_mean = results_df['cv_score_mean'].iloc[0]
                    cv_std = results_df['cv_score_std'].iloc[0] if 'cv_score_std' in results_df.columns else None
                    if cv_std is not None:
                        report_lines.append(f"- **CV score**: {cv_mean:.4f} ± {cv_std:.4f}")
                    else:
                        report_lines.append(f"- **CV score**: {cv_mean:.4f}")
            
            report_lines.append("")
            
            # Model hyperparameters
            if hasattr(pipeline.best_model_, 'get_params'):
                params = pipeline.best_model_.get_params()
                report_lines.append("### Hyperparameters")
                
                # Report all parameters
                for param, value in params.items():
                    report_lines.append(f"- `{param}`: {value}")
                
                report_lines.append("")
            
            # Feature importance (if available)
            if hasattr(pipeline.best_model_, 'feature_importances_'):
                report_lines.append("### Feature Importance")
                feature_importances = pipeline.best_model_.feature_importances_
                
                # Get feature names for importance
                if pipeline.transformed_feature_names_ and len(pipeline.transformed_feature_names_) == len(feature_importances):
                    feature_names_for_importance = pipeline.transformed_feature_names_
                else:
                    feature_names_for_importance = [f'Feature_{i}' for i in range(len(feature_importances))]
                
                # Sort by importance
                importance_pairs = list(zip(feature_names_for_importance, feature_importances))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 10 most important features
                report_lines.append("**Top 10 Most Important Features:**")
                for i, (feature_name, importance) in enumerate(importance_pairs[:10]):
                    report_lines.append(f"{i+1}. `{feature_name}`: {importance:.4f}")
                
                if len(importance_pairs) > 10:
                    report_lines.append(f"\n*... and {len(importance_pairs) - 10} more features*")
                
                report_lines.append("")

            # If test data is available, add test set analysis
            if X_test is not None and y_test is not None:                     
                
                report_lines.append("## Test Set Analysis")

                # Make predition
                y_pred = pipeline.predict(pipeline.X_test_original_)

                # If we are doing a classification task, make the right plots
                if pipeline.task_type == 'classification':
                    
                    # Get labels
                    labels = pipeline.labels_ if pipeline.labels_ else range(len(np.unique(y_train)))
                    
                    # Plot confusion matrix
                    with hush():
                        confusion_matrix_figure = plot_confusion_matrix(y_test, y_pred, labels=pipeline.label_mapping_)
                    confusion_matrix_path = self._save_plot(confusion_matrix_figure, f"confusion_matrix_v{version}", report_subfolder)
                    report_lines.append("### Confusion Matrix")
                    report_lines.append(f"![Confusion Matrix]({confusion_matrix_path})")
                    report_lines.append("")

                    # Plot ROC curve
                    y_test_proba = pipeline.predict_proba(pipeline.X_test_original_)
                    roc_curve_figure = plot_ROC_curve(y_test, y_test_proba, labels=labels)
                    roc_curve_path = self._save_plot(roc_curve_figure, f"roc_curve_v{version}", report_subfolder)
                    report_lines.append("### ROC Curve")
                    report_lines.append(f"![ROC Curve]({roc_curve_path})")

                elif pipeline.task_type == 'regression':

                    # Plot regression prediction plot
                    regression_prediction_figure = regression_prediction_plot(y_test, y_pred)
                    regression_prediction_path = self._save_plot(regression_prediction_figure, f"regression_prediction_v{version}", report_subfolder)
                    report_lines.append("### Regression Prediction Plot")
                    report_lines.append(f"![Regression Prediction]({regression_prediction_path})")

                    # Plot regression residuals
                    regression_residual_figure = regression_residual_plot(y_test, y_pred)
                    regression_residual_path = self._save_plot(regression_residual_figure, f"regression_residuals_v{version}", report_subfolder)
                    report_lines.append("### Regression Residuals Plot")
                    report_lines.append(f"![Regression Residuals]({regression_residual_path})")
                    
                      
        # Write the report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return report_path
    
    def _get_next_version(self):
        """Get the next version number for reports."""
        if not os.path.exists(self.report_directory):
            return "0001"

        # Look for existing report subfolders to determine next version
        pattern = re.compile(r"data_report_v(\d{4})_\d{8}_\d{6}$")
        version = 1

        # Check both files (old format) and directories (new format)
        for item in os.listdir(self.report_directory):
            item_path = os.path.join(self.report_directory, item)

            # Check directories (new format)
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    v = int(match.group(1))
                    version = max(version, v + 1)

            # Also check old format files for backward compatibility
            elif item.endswith('.md'):
                old_pattern = re.compile(r"data_report_v(\d{4})_.*\.md$")
                match = old_pattern.match(item)
                if match:
                    v = int(match.group(1))
                    version = max(version, v + 1)

        # Return zero-padded version
        return f"{version:04d}"