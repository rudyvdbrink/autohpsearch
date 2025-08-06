# %%

import unittest
from datasets import Dataset, ClassLabel
from autohpsearch.models.llms import AutoLoraForSeqClass

class TestAutoLoraForSeqClass(unittest.TestCase):
    def setUp(self):
        """Set up the dataset and label mapping for testing."""
        self.data = {
            "text": [
                "The app keeps crashing, it's so frustrating.",
                "The hotel room was clean and spacious, very comfortable.",
                "The delivery was delayed and the package arrived damaged.",
                "I enjoyed the presentation, it was very informative.",
                "The game was buggy and unplayable, I want a refund.",
                "The staff at the store were friendly and helpful.",
                "The movie was boring and way too long.",
                "I love this new phone, the camera quality is amazing.",
                "The software is outdated and doesn't work properly.",
                "The beach was beautiful and the water was crystal clear.",
                "The flight was delayed for hours, it was a nightmare.",
                "The coffee shop had great ambiance and delicious drinks.",
                "The car broke down after just a week of use.",
                "The park was peaceful and perfect for a morning walk."
            ],
            "label": [
                "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative",
                "Positive", "Negative", "Positive", "Positive", "Negative", "Positive"
            ]
        }

        self.label_mapping = {"Positive": 1, "Negative": 0}
        self.data["label"] = [self.label_mapping[label] for label in self.data["label"]]

        # Convert the data into a Hugging Face Dataset
        self.dataset = Dataset.from_dict(self.data)

        # Convert the label column to ClassLabel for binary classification
        num_classes = len(self.label_mapping)
        class_label = ClassLabel(num_classes=num_classes, names=list(self.label_mapping.keys()))
        self.dataset = self.dataset.cast_column("label", class_label)

        # Split the dataset into train+validation and test sets
        self.train_test_split = self.dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        base_model = "bert-base-uncased"

        # Initialize the model
        model = AutoLoraForSeqClass(
            base_model=base_model,
            r=2,
            train_batch_size=2,
            eval_batch_size=2,
            num_train_epochs=6,
        )

        # Check that the model is initialized correctly
        self.assertIsNotNone(model)

    def test_model_fit(self):
        """Test that the model's fit method works without errors."""
        base_model = "bert-base-uncased"

        # Initialize the model
        model = AutoLoraForSeqClass(
            base_model=base_model,
            r=2,
            train_batch_size=2,
            eval_batch_size=2,
            num_train_epochs=6,
        )

        # Fit the model on the dataset
        try:
            model.fit(self.train_test_split)
        except Exception as e:
            self.fail(f"Model fit method raised an exception: {e}")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)