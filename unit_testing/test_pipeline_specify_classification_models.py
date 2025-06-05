import unittest
import pandas as pd
import numpy as np
from autohpsearch.pipeline.pipeline import AutoMLPipeline

class TestPipelineSpecificModels(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.choice(['A', 'B', 'C'], size=100)
        })
        self.y_train = pd.Series(np.random.choice([0, 1], size=100))  # Binary target variable

        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'feature3': np.random.choice(['A', 'B', 'C'], size=50)
        })
        self.y_test = pd.Series(np.random.choice([0, 1], size=50))  # Binary target variable

        # Models to test
        self.models = [
            'logistic_regression',
            'random_forest_clf',
            'svm_clf',
            'gradient_boosting_clf',
            'knn_clf',
            'xgboost_clf',
            'dnn_clf'
        ]

        # Initialize the pipeline with specific models
        self.pipeline = AutoMLPipeline(
            search_type='random',
            model_name=self.models
        )

    def test_specific_models_in_pipeline(self):
        """Test that the pipeline uses only the specified models."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Check that the pipeline recognizes the specified models
        used_models = self.pipeline.model_name
        self.assertEqual(set(used_models), set(self.models), "Pipeline did not use the specified models.")

if __name__ == '__main__':
    unittest.main()