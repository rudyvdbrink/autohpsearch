import os
import glob
import unittest
import pandas as pd
import numpy as np
from autohpsearch.pipeline.pipeline import AutoMLPipeline

class TestPipelineSaveLoad(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.choice(['A', 'B', 'C'], size=100)
        })
        self.y_train = pd.Series(np.random.choice(['A', 'B'], size=100))  # Binary target variable

        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'feature3': np.random.choice(['A', 'B', 'C'], size=50)
        })
        self.y_test = pd.Series(np.random.choice(['A', 'B'], size=50))  # Binary target variable

        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            search_type='random',
            model_name=['logistic_regression', 'random_forest_clf'],
        )

        # Temporary file for saving the pipeline
        self.temp_file = "test_pipeline"

    def tearDown(self):
        # Clean up the temporary file after the test
        if os.path.exists(self.full_path[0]):
            os.remove(self.full_path[0])

    def test_save_and_load_pipeline(self):
        """Test saving and loading the pipeline."""
        # Fit the pipeline
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)

        # Make a prediction before saving
        self.pre_save_prediction = self.pipeline.predict(self.X_test)

        # Save the pipeline
        self.pipeline.save(filename=self.temp_file)

        # Look for saved file
        self.full_path = glob.glob(os.path.join('models', f"{self.temp_file}_v*.joblib"))

        # Check of the file exists
        self.assertTrue(len(self.full_path) > 0, "Pipeline file not found after saving.")

        # Load the pipeline
        loaded_pipeline = AutoMLPipeline.load(self.full_path[0])

        # Check that the loaded pipeline has the same models
        self.assertEqual(set(loaded_pipeline.model_name), set(self.pipeline.model_name), "Loaded pipeline models do not match the original pipeline.")

        # Check that the loaded pipeline can make predictions
        self.post_save_predictions = loaded_pipeline.predict(self.X_test)

        self.assertEqual(len(self.post_save_predictions), len(self.y_test), "Loaded pipeline failed to make predictions.")

        # Check that predictions before and after saving are the same
        np.testing.assert_array_equal(self.pre_save_prediction, self.post_save_predictions, "Predictions before and after saving the pipeline do not match.")

if __name__ == '__main__':
    unittest.main()