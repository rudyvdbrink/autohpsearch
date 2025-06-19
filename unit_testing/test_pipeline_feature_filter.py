import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autohpsearch.pipeline.pipeline import AutoMLPipeline
from sklearn.datasets import make_classification


class TestAutoMLPipeline(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        np.random.seed(42)
        # Generate synthetic data
        X, y = make_classification(
            n_samples=100,  # Number of samples
            n_features=5,   # Number of features
            n_informative=3,  # Number of informative features
            n_redundant=1,  # Number of redundant features
            n_classes=3,    # Number of classes
            random_state=42
        )

        # Convert to pandas DataFrame and Series
        X = pd.DataFrame(X, columns=[f'feature{i}' for i in range(1, 6)])
        y = pd.Series(y, name='target')

        # Duplicate one of the columns in X (to ensure feature filtering works)
        X['feature_duplicate'] = X['feature1']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            task_type='classification',
            cv=2,
            filter_features=True,
            model_name=['random_forest_clf', 'gradient_boosting_clf'],        
        )

    def test_pipeline_fit(self):
        """Test pipeline fitting."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(hasattr(self.pipeline, 'best_model_'), "Pipeline did not fit properly.")

    def test_generate_data_report(self):
        """Test the generate_data_report method."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        report = self.pipeline.generate_data_report()
        self.assertIsNotNone(report, "Data report was not generated.")

if __name__ == '__main__':
    unittest.main()