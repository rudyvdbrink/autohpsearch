import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autohpsearch.pipeline.pipeline import AutoMLPipeline

class TestPipelineSMOTEVariants(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def _run_pipeline(self, X, y, smote_kwargs=None):
        # Only use one classifier to keep test fast and simple
        pipeline = AutoMLPipeline(
            task_type='classification',
            model_name=['random_forest_clf'],
            apply_smote=True,
            smote_kwargs=smote_kwargs,
            verbose=True
        )
        # Split to train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
       
        # Run fit with SMOTE
        pipeline.fit(X_train, y_train, X_test, y_test)
        # Check class balance in y_train_processed_
        yb = pipeline.y_train_processed_
        values, counts = np.unique(yb, return_counts=True)
        self.assertTrue(np.all(counts == counts[0]), f"Classes not balanced after SMOTE: {counts}")
        # Check 3 classes present
        self.assertEqual(len(values), 3)

        # write a report
        pipeline.generate_data_report()

        return pipeline

    def test_smote_numeric(self):
        # All numeric features
        n = 120
        X = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.randn(n) + np.array([0.1 if i < 70 else 0.5 if i < 100 else 0.9 for i in range(n)]),  # Add moderate correlation to target
        })
        # 3-class imbalanced target (SMOTE, numeric)
        y = np.array([0]*70 + [1]*30 + [2]*20)

        # Run pipeline
        self._run_pipeline(X, y, smote_kwargs={'random_state': 42})

    def test_smote_categorical(self):
        # All categorical features
        n = 120
        X = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C', 'D'], size=n),
            'cat2': np.random.choice(['X', 'Y'], size=n)
        })
        y = np.array([0]*75 + [1]*25 + [2]*20)

        # Add moderate correlation to target by mapping categorical features
        X['cat1'] = X['cat1'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
        X['cat2'] = X['cat2'].map({'X': 0, 'Y': 1})

        # Run pipeline
        self._run_pipeline(X, y, smote_kwargs={'random_state': 42})

    def test_smote_nc(self):
        # Mixed numeric and categorical features
        n = 150
        X = pd.DataFrame({
            'num1': np.random.randn(n) + np.array([0.1 if i < 90 else 0.5 if i < 130 else 0.9 for i in range(n)]),  # Add moderate correlation to target
            'cat1': np.random.choice(['red', 'green', 'blue'], size=n),
            'cat2': np.random.choice(['dog', 'cat'], size=n)
        })
        y = np.array([0]*90 + [1]*40 + [2]*20)

        # Add moderate correlation to target by mapping categorical features
        X['cat1'] = X['cat1'].map({'red': 0, 'green': 1, 'blue': 2})
        X['cat2'] = X['cat2'].map({'dog': 0, 'cat': 1})

        # Run pipeline
        self._run_pipeline(X, y, smote_kwargs={'random_state': 42})

if __name__ == '__main__':
    unittest.main()