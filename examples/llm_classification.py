# %% Libraries

from autohpsearch.models.llms import AutoLoraForSeqClass

from autohpsearch.datasets.dataloaders import fetch_imdb

# %% Get the data

dataset = fetch_imdb()

# %% Fit a model

# Initialize the model with the base model and LoRA parameters
model = AutoLoraForSeqClass(base_model='bert-base-uncased',
                            r=2,
                            train_batch_size=8,
                            eval_batch_size=8,
                            num_train_epochs=3,
                            )

# Fit the model on the dataset
model.fit(dataset)