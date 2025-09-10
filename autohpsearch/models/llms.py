# %% libraries

import os
import numpy as np

from transformers import DataCollatorWithPadding
from huggingface_hub import login
from datasets import DatasetDict

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

from autohpsearch.utils.context import hush
from tabulate import tabulate

# %% Set logging

import logging

# Suppress specific warnings related to PEFT
logging.getLogger("peft").setLevel(logging.ERROR)  # Suppress PEFT-related warnings
logging.getLogger("torch.utils.checkpoint").setLevel(logging.ERROR)  # Suppress gradient checkpointing warnings
    
# %% Base for LoRA-based models

class AutoLoraBase():
    """
    AutoLoraBase is a base class designed for fine-tuning pre-trained transformer models using 
    Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation). 
    This class provides common functionality and structure for building task-specific fine-tuning 
    classes, such as AutoLoraForSeqClass and AutoLoraForSeqReg.

    Attributes:
    ----------
    base_model : str
        The name or path of the pre-trained transformer model to be used as the base model.
    r : int
        The rank of the LoRA adaptation matrix, controlling the dimensionality of the low-rank 
        approximation.
    num_train_epochs : int
        The number of epochs to train the model.
    tokenizer_ : transformers.PreTrainedTokenizer
        The tokenizer associated with the base model, used for tokenizing input text.
    training_args_ : transformers.TrainingArguments
        The training arguments used to configure the training process, such as batch size, learning 
        rate, and evaluation strategy.
    train_dataset_ : datasets.Dataset
        The training dataset containing input text and labels for fine-tuning.
    test_dataset_ : datasets.Dataset
        The evaluation dataset used for validating the model during training.
    trainer_ : transformers.Trainer
        The Trainer object responsible for managing the training and evaluation process.
    training_results_ : dict
        The results of the training process, including metrics such as loss and task-specific 
        evaluation metrics.

    Methods:
    -------
    __init__(self, base_model, r, num_train_epochs, **kwargs):
        Initializes the AutoLoraBase object with the specified base model, LoRA rank, 
        number of training epochs, and additional arguments.

    fit(self, dataset):
        Fine-tunes the model on the provided dataset. Maps the dataset, configures the Trainer, 
        and trains the model.

    predict(self, dataset):
        Generates predictions for the given dataset using the fine-tuned model.

    evaluate(self, dataset):
        Evaluates the model on the provided dataset and returns evaluation metrics.

    push(self, save_name=None, private=True):
        Pushes the trained model to Hugging Face Hub, including the model, tokenizer, and training
        arguments. If `save_name` is not provided, it uses the model's save name.

    _map_dataset(self, dataset):
        Maps the input dataset to the format required by the model, including tokenization and 
        label processing.

    _set_target_modules(self):
        Dynamically sets the target modules for LoRA based on the base model type. This method is
        intended to be overridden by subclasses if specific target modules are required.

    _configure_trainer(self):
        Configures the Trainer object with the model, training arguments, datasets, and metrics.

    _compute_metrics(self, pred):
        Computes evaluation metrics for the model. This method is intended to be overridden by 
        subclasses to provide task-specific metrics.

    Notes:
    -----
    - This class serves as a base for task-specific fine-tuning classes and is not intended to be 
      used directly.
    - LoRA is applied to the model's attention layers to enable efficient fine-tuning with fewer 
      trainable parameters.
    - The class supports integration with Hugging Face's datasets and transformers libraries for 
      seamless data processing and model training.
    - Subclasses should override methods like `_compute_metrics` to implement task-specific 
      functionality.
    """
    def __init__(self, 
                 base_model="bert-base-uncased",
                 gradient_checkpointing=True, 
                 r=8, 
                 task_type=TaskType.SEQ_CLS,
                 train_batch_size=8,
                 eval_batch_size=8,
                 num_train_epochs=6,
                 metric_for_best_model=None,
                 target_modules=None,
                 gradient_accumulation_steps=4,
                 ):
        """
        Initialize the model with LoRA configuration.
        
        Args:
            base_model (str): The name of the pre-trained model.
            r (int): The low-rank dimension for LoRA.
            task_type (TaskType): The type of task for LoRA.
        """

        self.base_model = base_model
        self.model_name = base_model + "-lora"
        self.save_name = self.model_name.split('/')[-1]
        
        self.gradient_checkpointing = gradient_checkpointing  
        self.r = r  # Low-rank dimension for LoRA
        self.task_type = task_type  # Task type for LoRA
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.metric_for_best_model = metric_for_best_model
        self.target_modules_ = target_modules
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tokenizer_ = AutoTokenizer.from_pretrained(base_model)
        self.model = None  
        self.predictions = None      
        self._configure_lora()  # Configure LoRA settings
        self._configure_training()  # Configure training settings

    def _tokenize_function(self, examples):
        """
        Tokenize the input text and prepare the labels.
        Args:
            examples (dict): A dictionary containing the input text and labels.
        Returns:
            dict: A dictionary containing tokenized inputs and labels.
        """
        pass
    
    def _map_dataset(self,dataset):
        """
        Maps the input dataset to the format required by the model, including tokenization and
        label processing.
        Args:
            dataset (Dataset): The dataset to map.
        Returns:
            Dataset: The mapped dataset ready for training or evaluation.
        """
        dataset = dataset.map(self._tokenize_function, batched=True)
        dataset.set_format("torch")
        return dataset

    def _set_destination_dir(self, save_name=None):
        """
        Set the destination directory for saving the model.
        
        Args:
            save_name (str): The name to save the model under. If None, uses self.save_name.
        """
        if save_name is None:
            save_name = self.save_name
        
        return "./models/" + save_name
    
    def _set_target_modules(self):
        """
        Dynamically sets the target modules for LoRA based on the base model type.
        This method checks the base model's name and assigns the appropriate target modules
        for LoRA adaptation. It is intended to be overridden by subclasses if specific target
        modules are required.
        Raises:
            ValueError: If the base model type is unsupported.
        """

        # Dynamically set target_modules based on the base model type
        model_type = self.tokenizer_.name_or_path.split("-")[0]  # Extract model type from base model name

        if "distilbert" in model_type:
            target_modules = ["q_lin", "k_lin"]
        elif "bert" in model_type:
            target_modules = ["query", "key", "value"]
        elif "gpt" in model_type:
            target_modules = ["c_attn"]
        elif "llama" in model_type or 'meta' in model_type:
            target_modules = ["q_proj", "k_proj"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.target_modules_ = target_modules
    
    def _configure_lora(self):   
        """
        Configure LoRA (Low-Rank Adaptation) settings for the model.
        This method initializes the LoRA configuration with the specified task type, low-rank
        dimension, and target modules. 
        Raises:
            ValueError: If the target modules are not set or if the task type is unsupported.
        """

        if self.target_modules_ == 'auto':
            self._set_target_modules()     
        
        # Configure LoRA
        self.lora_config_ = LoraConfig(
                            task_type=self.task_type,  
                            inference_mode=False,
                            r=self.r,  # Low-rank dimension
                            lora_alpha=32,
                            lora_dropout=0.1,
                            target_modules=self.target_modules_  # Apply LoRA only to the query, key and value layers
                        )
    
    def _configure_training(self):
        """
        Configure training parameters for the model.
        This method sets up the training arguments, including output directory, evaluation strategy,
        learning rate, batch sizes, number of epochs, weight decay, logging settings, and other
        training-related parameters. It also sets the destination directory for saving the model.
        Raises:
            ValueError: If the metric for best model is not specified.
        """

        # Set the destination directory for saving the model
        output_dir = self._set_destination_dir()
        
        #print("Model checkpoints stored in: " + output_dir)

        # Configure training parameters
        self.training_args_ = TrainingArguments(
                            output_dir=output_dir,
                            eval_strategy="epoch",  
                            save_strategy="epoch",
                            learning_rate=2e-5,
                            per_device_train_batch_size=self.train_batch_size,
                            per_device_eval_batch_size=self.eval_batch_size,
                            num_train_epochs=self.num_train_epochs,
                            weight_decay=0.01,
                            logging_dir="./logs",
                            logging_steps=10,
                            load_best_model_at_end=True,
                            metric_for_best_model=self.metric_for_best_model,
                            save_total_limit=2,
                            report_to="none",
                            gradient_accumulation_steps=4, # Accumulate gradients over 4 steps                            
                            fp16=True,  # Enable mixed precision
                        )
        
    def _configure_trainer(self):
        """
        Configure the Trainer object with the model, training arguments, datasets, and metrics.
        This method initializes the Trainer with the model, training arguments, training and evaluation datasets,
        a data collator for dynamic padding, and a compute metrics function for evaluation.
        Raises:
            ValueError: If the model or tokenizer is not set.
        """
        # Use a data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer_)
        
        self.trainer_ = Trainer(
            model=self.model,
            args=self.training_args_,
            train_dataset=self.train_dataset_,
            eval_dataset=self.test_dataset_,
            data_collator=data_collator,  # Use data_collator instead of tokenizer
            compute_metrics=self._compute_metrics,            
        )
    
    def fit(self, dataset):
        """
        Fine-tunes the model on the provided dataset using LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning).

        Args:
            dataset (dict): A dictionary containing the training and test datasets. 
                - Keys: 'train' and 'test'.
                - Each dataset must include features:
                    - 'text': The input text for the model.
                    - 'label': The corresponding labels for classification / regression.

        Raises:
            ValueError: If the dataset does not include both 'train' and 'test' keys or if the required features ('text' and 'label') are missing.

        Notes:
            - The method ensures the tokenizer has a padding token and resizes the model's token embeddings accordingly.
            - The model is set to training mode, and gradient checkpointing is enabled if specified.
            - The training and test datasets are mapped to the format required by the model.
            - After training, the model is evaluated, and the evaluation metrics are printed in a tabular format.
        """

        # Ensure the dataset is a DatasetDict with 'train' and 'test' splits
        if not isinstance(dataset, DatasetDict) or 'train' not in dataset or 'test' not in dataset:
            raise ValueError("Dataset must be a Huggingface Dataset with 'train' and 'test' splits.")

        # Get the number of unique labels from the dataset
        self._get_classes(dataset)

        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=self.num_classes)

        # Apply PEFT with LoRA to the model
        self.model = get_peft_model(self.model, self.lora_config_)

        # set model to training mode
        self.model.train()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        # Ensure the tokenizer has a padding token
        if self.tokenizer_.pad_token is None:
            self.tokenizer_.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer_))  # Resize embeddings to include the new token
            self.model.config.pad_token_id = self.tokenizer_.pad_token_id

        self.train_dataset_ = dataset['train']
        self.test_dataset_  = dataset['test']

        self.train_dataset_ = self._map_dataset(self.train_dataset_)
        self.test_dataset_  = self._map_dataset(self.test_dataset_)

        self._configure_trainer()

        self.trainer_.train()

        # Evaluate the model
        self.training_results_ = self.trainer_.evaluate()
        table = tabulate(self.training_results_.items(), headers=["Metric", "Value"], tablefmt="simple_outline")
        print(table)        
    
    def push(self, save_name=None, private=True):
        """
        Push the trained model to Hugging Face Hub.
        
        Args:
            save_name (str): The name to save the model under. If None, uses self.save_name.
        """
        if save_name is None:
            save_name = self.save_name

        # get hugging face API key from environment variable
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY not found. Set it in your environment or .env file.")

        # log into Hugging Face
        login(token=api_key)
        
        # Push the trained model to Hugging Face
        self.model.push_to_hub(save_name, private=private)

        # Optionally, push the tokenizer and trainer arguments as well
        self.tokenizer_.push_to_hub(save_name, private=private)

        # Push the training arguments using the Trainer
        self.trainer_.push_to_hub(save_name)

        print(f"Model pushed to Hugging Face Hub: https://huggingface.co/{save_name}")

# %% Classification

class AutoLoraForSeqClass(AutoLoraBase):
    """
    AutoLoraForSeqClass is a class designed for fine-tuning sequence classification models using 
    Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation). 
    This class simplifies the process of applying LoRA to pre-trained transformer models for 
    sequence classification tasks.

    Methods:
    -------
    __init__(self, base_model, r, num_train_epochs, **kwargs):
        Initializes the AutoLoraForSeqClass object with the specified base model, LoRA rank, 
        number of training epochs, and additional arguments.

    fit(self, dataset):
        Fine-tunes the model on the provided dataset. Maps the dataset, configures the Trainer, 
        and trains the model.

    predict(self, dataset):
        Generates predictions for the given dataset using the fine-tuned model.

    evaluate(self, dataset):
        Evaluates the model on the provided dataset and returns evaluation metrics.
    
    push(self, save_name=None, private=True):
        Pushes the trained model to Hugging Face Hub, including the model, tokenizer, and training
        arguments. If `save_name` is not provided, it uses the model's save name.

    _map_dataset(self, dataset):
        Maps the input dataset to the format required by the model, including tokenization and 
        label processing.
    
    _set_target_modules(self):
        Dynamically sets the target modules for LoRA based on the base model type. This method is
        intended to be overridden by subclasses if specific target modules are required.

    _configure_trainer(self):
        Configures the Trainer object with the model, training arguments, datasets, and metrics.

    _compute_metrics(self, pred):
        Computes evaluation metrics for the model, such as accuracy, precision, recall, and F1 
        score.
    
    Attributes:
    ----------
    base_model : str
        The name or path of the pre-trained transformer model to be used as the base model.
    r : int
        The rank of the LoRA adaptation matrix, controlling the dimensionality of the low-rank 
        approximation.
    num_train_epochs : int
        The number of epochs to train the model.
    tokenizer_ : transformers.PreTrainedTokenizer
        The tokenizer associated with the base model, used for tokenizing input text.
    training_args_ : transformers.TrainingArguments
        The training arguments used to configure the training process, such as batch size, learning 
        rate, and evaluation strategy.
    train_dataset_ : datasets.Dataset
        The training dataset containing input text and labels for fine-tuning.
    test_dataset_ : datasets.Dataset
        The evaluation dataset used for validating the model during training.
    trainer_ : transformers.Trainer
        The Trainer object responsible for managing the training and evaluation process.
    training_results_ : dict
        The results of the training process, including metrics such as accuracy and loss.

    Notes:
    -----
    - This class is specifically designed for sequence classification tasks and assumes that the 
      base model is compatible with the transformers library.
    - LoRA is applied to the model's attention layers to enable efficient fine-tuning with fewer 
      trainable parameters.
    - The class supports integration with Hugging Face's datasets and transformers libraries for 
      seamless data processing and model training.
    """
    def __init__(self, 
                 metric_for_best_model="accuracy",
                 **kwargs):
        """
        Initialize the AutoLoraForSeqClass object with the specified base model, LoRA rank,
        number of training epochs, and additional arguments.
        Args:
            base_model (str): The name of the pre-trained model.
            r (int): The low-rank dimension for LoRA.
            task_type (TaskType): The type of task for LoRA.
        """
        super().__init__(metric_for_best_model=metric_for_best_model,
                         **kwargs)       
        

    def _set_destination_dir(self):
        """
        Set the destination directory for saving the model.
        
        Args:
            save_name (str): The name to save the model under. If None, uses self.save_name.
        """
        self.model_name = self.model_name + "-classifier"
        self.save_name = self.model_name.split('/')[-1] 
        
        return "./models/" + self.save_name
    
    def _tokenize_function(self, examples):
        """
        Tokenize the input text and prepare the labels for classification.
        Args:
            examples (dict): A dictionary containing the input text and labels.
        Returns:
            dict: A dictionary containing tokenized inputs and labels.
        """

        return self.tokenizer_(examples['text'], truncation=True)

    def _compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.
        Args:
            eval_pred (tuple): A tuple containing logits and labels.
        Returns:
            dict: A dictionary containing accuracy, balanced accuracy, F1 score, precision, and recall.
        """

        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        acc = accuracy_score(labels, predictions)
        bacc = balanced_accuracy_score(labels, predictions)
        
        return {"accuracy": acc, "balanced accuracy": bacc, "f1": f1, "precision": precision, "recall": recall}

    def _get_classes(self, dataset):
        """
        Get the number of unique classes from the dataset.
        Args:
            dataset (DatasetDict): The dataset containing 'train' split.
        """

        self.num_classes = len(set(dataset['train']['label']))

    def predict_proba(self, dataset):
        """
        Predict class probabilities for a given dataset.
        
        Args:
            dataset (Dataset): The dataset to predict on.
        
        Returns:
            List: Predicted probabilities.
        """

        # Map the dataset
        dataset = self._map_dataset(dataset)

        # Use the trainer to predict
        self.predictions = self.trainer_.predict(dataset)

        logits = self.predictions.predictions  # Extract logits
        self.probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return self.probabilities

    def predict(self, dataset):
        """
        Predict labels for a given dataset.
        
        Args:
            dataset (Dataset): The dataset to predict on.
        
        Returns:
            Array: Predicted labels.
        """

        _ = self.predict_proba(dataset)

        return np.argmax(self.predictions.predictions, axis=1)


# %% Regression

# Custom data collator to ensure labels are floats
class RegressionDataCollator(DataCollatorWithPadding):
    """
    RegressionDataCollator is a custom data collator designed for regression tasks. It ensures 
    that input features and labels are dynamically padded and that labels are cast to the 
    appropriate data type (e.g., torch.float) for regression models.

    This class extends the functionality of Hugging Face's `DataCollatorWithPadding` to handle 
    regression-specific requirements, such as ensuring labels are in the correct format for 
    loss computation.

    Attributes:
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model, used for dynamic padding of input features.

    Methods:
    -------
    __call__(self, features):
        Processes a batch of features, dynamically pads input features, and ensures labels are 
        cast to torch.float for regression tasks.

    Notes:
    -----
    - This data collator is specifically designed for regression tasks and assumes that the 
      labels are numerical values.
    - It integrates seamlessly with Hugging Face's `Trainer` class for batching and padding 
      during training and evaluation.
    - Labels are cast to torch.float to ensure compatibility with regression loss functions 
      such as `MSELoss`.
    """

    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()  # Ensure labels are floats
        return batch

class AutoLoraForSeqReg(AutoLoraBase):
    """
    AutoLoraForSeqReg is a class designed for fine-tuning regression models using 
    Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation). 
    This class simplifies the process of applying LoRA to pre-trained transformer models for 
    regression tasks.

    Attributes:
    ----------
    base_model : str
        The name or path of the pre-trained transformer model to be used as the base model.
    r : int
        The rank of the LoRA adaptation matrix, controlling the dimensionality of the low-rank 
        approximation.
    num_train_epochs : int
        The number of epochs to train the model.
    tokenizer_ : transformers.PreTrainedTokenizer
        The tokenizer associated with the base model, used for tokenizing input text.
    training_args_ : transformers.TrainingArguments
        The training arguments used to configure the training process, such as batch size, learning 
        rate, and evaluation strategy.
    train_dataset_ : datasets.Dataset
        The training dataset containing input text and labels for fine-tuning.
    test_dataset_ : datasets.Dataset
        The evaluation dataset used for validating the model during training.
    trainer_ : transformers.Trainer
        The Trainer object responsible for managing the training and evaluation process.
    training_results_ : dict
        The results of the training process, including metrics such as mean squared error (MSE), 
        root mean squared error (RMSE), and R-squared.

    Methods:
    -------
    __init__(self, base_model, r, num_train_epochs, **kwargs):
        Initializes the AutoLoraForSeqReg object with the specified base model, LoRA rank, 
        number of training epochs, and additional arguments.

    fit(self, dataset):
        Fine-tunes the model on the provided dataset. Maps the dataset, configures the Trainer, 
        and trains the model.

    predict(self, dataset):
        Generates predictions for the given dataset using the fine-tuned model.

    evaluate(self, dataset):
        Evaluates the model on the provided dataset and returns evaluation metrics.
    
    push(self, save_name=None, private=True):
        Pushes the trained model to Hugging Face Hub, including the model, tokenizer, and training
        arguments. If `save_name` is not provided, it uses the model's save name.

    _map_dataset(self, dataset):
        Maps the input dataset to the format required by the model, including tokenization and 
        label processing.
    
    _set_target_modules(self):
        Dynamically sets the target modules for LoRA based on the base model type. This method is
        intended to be overridden by subclasses if specific target modules are required.

    _configure_trainer(self):
        Configures the Trainer object with the model, training arguments, datasets, and metrics.

    _compute_metrics(self, pred):
        Computes evaluation metrics for the model, such as mean squared error (MSE), mean absolute 
        error (MAE), root mean squared error (RMSE), and R-squared.

    Notes:
    -----
    - This class is specifically designed for regression tasks and assumes that the base model is 
      compatible with the transformers library.
    - LoRA is applied to the model's attention layers to enable efficient fine-tuning with fewer 
      trainable parameters.
    - The class supports integration with Hugging Face's datasets and transformers libraries for 
      seamless data processing and model training.
    - The model's output layer is configured to produce a single continuous value per input, making 
      it suitable for regression tasks.
    """
    def __init__(self, 
                 metric_for_best_model="rmse",
                 **kwargs):
        super().__init__(metric_for_best_model=metric_for_best_model,
                         **kwargs)
        
        """
        Initialize the AutoLoraForSeqReg object with the specified base model, LoRA rank,
        number of training epochs, and additional arguments.
        
        Args:
            base_model (str): The name of the pre-trained model.
            r (int): The low-rank dimension for LoRA.
            task_type (TaskType): The type of task for LoRA.
        """
        
    def _set_destination_dir(self):
            """
            Set the destination directory for saving the model.
            
            Args:
                save_name (str): The name to save the model under. If None, uses self.save_name.
            """

            self.model_name = self.model_name + "-regressor"
            self.save_name = self.model_name.split('/')[-1] 
            
            return "./models/" + self.save_name

    def _tokenize_function(self, examples):
        """
        Tokenize the input text and prepare the labels for regression.
        Args:
            examples (dict): A dictionary containing the input text and labels.
        Returns:
            dict: A dictionary containing tokenized inputs and labels.
        """
        tokenized = self.tokenizer_(examples['text'], truncation=True)
        tokenized["label"] = [float(label) for label in examples["label"]]  # Convert labels to floats
        return tokenized
    
    def _configure_trainer(self):
        """
        Configure the Trainer object for regression tasks.
        This method sets up the Trainer with the model, training arguments, datasets, and a custom
        data collator for regression tasks.
        """

        # Use a data collator for dynamic padding
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer_)
        data_collator = RegressionDataCollator(tokenizer=self.tokenizer_)
        
        self.trainer_ = Trainer(
            model=self.model,
            args=self.training_args_,
            train_dataset=self.train_dataset_,
            eval_dataset=self.test_dataset_,
            data_collator=data_collator,  # Use data_collator instead of tokenizer
            compute_metrics=self._compute_metrics,            
        )
    
    def _compute_metrics(self, pred):
        """
        Compute metrics for regression evaluation.
        Args:
            pred (tuple): A tuple containing predictions and labels.
        Returns:
            dict: A dictionary containing evaluation metrics such as MSE, MAE, R-squared, and RMSE.
        """
        predictions, labels = pred
        predictions = predictions.flatten()  # Flatten predictions for regression
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)
        return {"mse": mse, "mae": mae, "r2": r2, "rmse": rmse}

    def _get_classes(self, dataset):
        """
        Just sets the number of classes to 1, which is appropriate for regression tasks.
        Args:
            dataset (DatasetDict): The dataset containing 'train' split.
        """

        self.num_classes = 1        

    def predict(self, dataset):
        """
        Predict labels for a given dataset.
        
        Args:
            dataset (Dataset): The dataset to predict on.
        
        Returns:
            Array: Predicted labels.
        """
        # Map the dataset
        dataset = self._map_dataset(dataset)

        # Use the trainer to predict
        self.predictions = self.trainer_.predict(dataset)
        
        # Get probability
        preds = self.predictions.predictions.flatten()
        
        return preds

