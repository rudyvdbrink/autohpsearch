import unittest
import numpy as np
import pandas as pd
from autohpsearch import AutoMLPipeline

class TestAutoMLPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5],  
            'C': [5, 4, 3, 2, 1],  
            'D': [1, 1, 1, 1, 2],   
        })
        self.y = np.random.choice([0, 1], size=5)

        # Duplicate X and y
        self.X = pd.concat([self.X] * 20, ignore_index=True)
        self.y = np.concatenate([self.y] * 20)

        # #convert y to a pandas Series
        self.y  = pd.Series(self.y)
        

    def test_pipeline_df(self):
        """Test the AutoMLPipeline functionality."""
        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            cv=2,
            model_name=['gradient_boosting_clf'],
            drop_duplicate_rows=True,
        )

        # Fit the pipeline
        self.pipeline.fit(self.X, self.y, self.X, self.y)

        # Check if duplicate rows were dropped
        self.assertLess(len(self.pipeline.X_train_processed_), len(self.X), "Duplicate rows were not dropped.")

        # Make a report
        self.report = self.pipeline.generate_data_report()

        # Check if the data report was generated
        self.assertIsNotNone(self.report, "Report was not generated.")

    def test_pipeline_np(self):
        """Test the AutoMLPipeline functionality."""
        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            cv=2,
            model_name=['gradient_boosting_clf'],
            drop_duplicate_rows=True,
        )

        # Convert X to numpy array
        self.X = self.X.to_numpy()
        # Convert y to numpy array
        self.y = self.y.to_numpy()

        # Fit the pipeline
        self.pipeline.fit(self.X, self.y, self.X, self.y)

        # Check if duplicate rows were dropped
        self.assertLess(len(self.pipeline.X_train_processed_), len(self.X), "Duplicate rows were not dropped.")

        # Make a report
        self.report = self.pipeline.generate_data_report()

        # Check if the data report was generated
        self.assertIsNotNone(self.report, "Report was not generated.")

if __name__ == '__main__':
    unittest.main()