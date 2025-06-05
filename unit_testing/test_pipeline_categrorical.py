import unittest
import pandas as pd
import numpy as np
from autohpsearch.pipeline.pipeline import AutoMLPipeline

class TestPipelineCategoricalY(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.choice(['A', 'B', 'C'], size=100)
        })
        self.y_train = pd.Series(np.random.choice(['class1', 'class2', 'class3'], size=100))  # Categorical target variable

        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'feature3': np.random.choice(['A', 'B', 'C'], size=50)
        })
        self.y_test = pd.Series(np.random.choice(['class1', 'class2', 'class3'], size=50))  # Categorical target variable

        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            task_type='classification',
            model_name=['random_forest_clf', 
                        'gradient_boosting_clf',
                        'xgboost_clf'],
        )

    def test_generate_data_report_with_categorical_y(self):
        """Test generate_data_report with a categorical y variable."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        try:
            report = self.pipeline.generate_data_report()
            self.assertIsNotNone(report, "Data report was not generated.")
        except Exception as e:
            self.fail(f"generate_data_report raised an exception with categorical y: {e}")

if __name__ == '__main__':
    unittest.main()