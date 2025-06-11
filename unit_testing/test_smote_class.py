import unittest
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

# Make sure to import your SMOTEApplier from its location:
from autohpsearch.pipeline.cleaning import SMOTEApplier

class MockPipeline:
    """A minimal pipeline mock for supplying feature lists."""
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features_ = numerical_features
        self.categorical_features_ = categorical_features

class TestSMOTEApplierVariants(unittest.TestCase):
    def test_smote_numeric(self):
        # All numeric features (triggers plain SMOTE)
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
        })
        y = np.array([0]*90 + [1]*10)  # Highly imbalanced

        pipeline = MockPipeline(numerical_features=['num1', 'num2'], categorical_features=[])
        smote = SMOTEApplier(pipeline=pipeline, random_state=42)
        X_res, y_res = smote.fit_transform(X, y)

        # After SMOTE, classes should be balanced
        _, counts = np.unique(y_res, return_counts=True)
        self.assertTrue(np.all(counts == counts[0]))
        # Ensure original shape increased
        self.assertGreater(X_res.shape[0], X.shape[0])

    def test_smote_categorical(self):
        # All categorical features (triggers SMOTEN)
        X = pd.DataFrame({
            'cat1': np.random.choice(['a', 'b', 'c'], size=100),
            'cat2': np.random.choice(['d', 'e'], size=100)
        })
        y = np.array([0]*80 + [1]*20)

        pipeline = MockPipeline(numerical_features=[], categorical_features=['cat1', 'cat2'])
        smote = SMOTEApplier(pipeline=pipeline, random_state=42)
        X_res, y_res = smote.fit_transform(X, y)

        # Check shapes and balance
        _, counts = np.unique(y_res, return_counts=True)
        self.assertTrue(np.all(counts == counts[0]))
        self.assertGreater(X_res.shape[0], X.shape[0])

    def test_smote_nc(self):
        # Mixed numeric and categorical (triggers SMOTENC)
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'cat1': np.random.choice(['a', 'b'], size=100)
        })
        y = np.array([0]*70 + [1]*30)

        pipeline = MockPipeline(numerical_features=['num1'], categorical_features=['cat1'])
        smote = SMOTEApplier(pipeline=pipeline, random_state=42)
        X_res, y_res = smote.fit_transform(X, y)

        # Check balance
        _, counts = np.unique(y_res, return_counts=True)
        self.assertTrue(np.all(counts == counts[0]))
        # Ensure types preserved: cat1 should still be present
        self.assertIn('cat1', X_res.columns)
        self.assertIn('num1', X_res.columns)

if __name__ == '__main__':
    unittest.main()