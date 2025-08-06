#  %% libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict

# %% functions 

def fetch_housing():
    """Fetch housing data, discard categorical features, and split into train/test sets."""
    data = fetch_openml(data_id=42165, as_frame=True)
    data = data.frame
    data.drop('Id', axis=1, inplace=True)
    target = data.pop('SalePrice')
    
    # Discard categorical features
    data = data.select_dtypes(include=['number'])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test


def fetch_imdb():

    # Load the IMDB dataset
    dataset = load_dataset("imdb")

    # Keep only the 'train' and 'test' splits
    dataset = DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"]
    })

    return dataset