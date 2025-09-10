# %% libraries

import numpy as np
from typing import Dict, Any, Optional
from inspect import signature

import torch
import torch.nn as nn

from datasets import Dataset, DatasetDict
from huggingface_hub import login

from transformers import (
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from autohpsearch.models.llms import AutoLoraBase  # Reuse base settings (tokenizer, args, logging)
from tabulate import tabulate


# %% Dual-head backbone + heads

class MultiTaskSequenceModel(nn.Module):
    """
    A dual-head model wrapping a transformer backbone (already PEFT-wrapped).
    - backbone: a transformer encoder (e.g., AutoModel), possibly wrapped with PEFT (LoRA)
    - classifier: produces logits for classification (num_labels)
    - regressor: produces a single continuous output

    Forward returns dict(logits=(logits_cls, logits_reg)) so Trainer can gather both.
    """

    def __init__(self, backbone: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        # Expose config to keep Trainer/Accelerate happy when they probe model.config
        self.config = getattr(self.backbone, "config", None)

        hidden_size = None
        if self.config is not None:
            hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone.config; please ensure a compatible encoder.")

        self.num_labels = num_labels
        if self.config is not None:
            self.config.num_labels = num_labels
            # Do NOT set use_return_dict here; some configs expose it as a read-only property.

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.regressor = nn.Linear(hidden_size, 1)

    def _pool(self, outputs) -> torch.Tensor:
        # Prefer pooled output if available; otherwise use CLS token
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return self.dropout(pooled)
        last_hidden = outputs.last_hidden_state  # [batch, seq, hidden]
        cls = last_hidden[:, 0, :]
        return self.dropout(cls)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        # Strip label fields if accidentally routed in
        kwargs = {k: v for k, v in kwargs.items() if k not in ("labels_cls", "labels_reg")}
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,  # enforce ModelOutput for compatibility
            **kwargs,
        )
        pooled = self._pool(outputs)
        logits_cls = self.classifier(pooled)             # [batch, num_labels]
        logits_reg = self.regressor(pooled).squeeze(-1)  # [batch]
        return {"logits": (logits_cls, logits_reg)}


# %% Custom Trainer computing joint loss

class DualHeadTrainer(Trainer):
    """
    Computes total loss = alpha_cls * CrossEntropy + alpha_reg * MSE.

    Expects input batches to include:
      - labels_cls: LongTensor of class indices
      - labels_reg: FloatTensor of regression targets
    """

    def __init__(self, alpha_cls: float = 1.0, alpha_reg: float = 1.0, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_cls = alpha_cls
        self.alpha_reg = alpha_reg
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss()  # default (unweighted)
        self.mse = nn.MSELoss()

    # Accept **kwargs for Transformers versions that pass extra args like num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_cls = inputs.pop("labels_cls", None)
        labels_reg = inputs.pop("labels_reg", None)

        outputs = model(**inputs)  # {"logits": (logits_cls, logits_reg)}
        logits_cls, logits_reg = outputs["logits"]

        loss = None
        if labels_cls is not None:
            if self.class_weights is not None:
                ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits_cls.device))
            else:
                ce = self.ce
            loss_cls = ce(logits_cls, labels_cls)
            loss = self.alpha_cls * loss_cls if loss is None else loss + self.alpha_cls * loss_cls

        if labels_reg is not None:
            loss_reg = self.mse(logits_reg, labels_reg.view(-1))
            loss = self.alpha_reg * loss_reg if loss is None else loss + self.alpha_reg * loss_reg

        return (loss, outputs) if return_outputs else loss


# %% Data collator enforcing label dtypes

class DualHeadDataCollator(DataCollatorWithPadding):
    """
    Ensures:
      - labels_cls -> torch.long
      - labels_reg -> torch.float
    """

    def __call__(self, features):
        batch = super().__call__(features)
        if "labels_cls" in batch:
            batch["labels_cls"] = batch["labels_cls"].long()
        if "labels_reg" in batch:
            batch["labels_reg"] = batch["labels_reg"].float()
        return batch


# %% Dual-head LoRA class

class AutoLoraForSeqDual(AutoLoraBase):
    """
    AutoLoraForSeqDual
    ==================

    A multitask (dual‑head) LoRA fine‑tuning wrapper for transformer encoders that jointly
    optimizes:
    1) A classification head (categorical / single‑label)
    2) A regression head (continuous target)

    It adapts only a small subset of the backbone parameters via LoRA adapters (PEFT) while
    training lightweight task heads on top of the shared representation. Both heads are trained
    simultaneously with a combined loss:

        total_loss = alpha_cls * CrossEntropy(logits_cls, labels_cls)
                + alpha_reg * MSE(pred_reg, labels_reg_normalized)

    Optionally, the regression targets can be normalized prior to training to keep the MSE term
    on a scale comparable to cross‑entropy, improving loss balance and gradient flow.

    Intended Dataset Format
    -----------------------
    A Hugging Face `DatasetDict` with at least the splits: 'train' and 'test' (or evaluation split).
    Each split must contain:
    - "text"       : str (single text input; extend manually if you need pair inputs)
    - "label_cls"  : int (class index, 0 .. num_classes-1)
    - "label_reg"  : float (continuous value)

    Normalization (optional) is applied BEFORE tokenization if `normalize_regression=True`.

    Key Behaviors
    -------------
    - LoRA is applied only to the transformer backbone (AutoModel), not to the custom heads.
    - Classification and regression heads share the backbone representation.
    - Loss weighting is controlled via alpha_cls and alpha_reg.
    - If regression normalization is enabled, inverse transformation is stored and applied to
    predicted regression outputs in `predict_reg` / `predict`.
    - Only LoRA adapters (not the dual heads) are pushed to the Hub via `push()`. Reconstructing
    the full multitask model downstream requires reinstantiating the class + loading adapters.

    Parameters
    ----------
    base_model : str, default="bert-base-uncased"
        Name or path of a pretrained Hugging Face transformer model (encoder-style is expected).
    gradient_checkpointing : bool, default=True
        Enable gradient checkpointing to reduce memory usage (slower compute).
    r : int, default=8
        Rank (intrinsic dimensionality) of LoRA adaptation matrices.
    task_type : peft.TaskType, default=TaskType.FEATURE_EXTRACTION (forced internally if not provided)
        PEFT task type. For encoder feature extraction + custom heads this should be FEATURE_EXTRACTION.
    train_batch_size : int, default=8
        Per-device batch size for training.
    eval_batch_size : int, default=8
        Per-device batch size for evaluation/prediction.
    num_train_epochs : int, default=6
        Number of training epochs.
    metric_for_best_model : str or None, default="accuracy"
        Name of a metric key returned by `_compute_metrics` to select and retain the best checkpoint.
        Examples: "accuracy", "f1", "balanced accuracy", "mse", "rmse". (Do not prefix with 'eval_'.)
    target_modules : list[str] | str | None, default=None
        List of module name substrings to which LoRA is applied. If set to "auto", a heuristic
        tries to infer suitable attention projection names. If None, you must provide manually
        or call with 'auto'.
    gradient_accumulation_steps : int, default=4
        Number of forward passes to accumulate before each optimizer step.
    alpha_cls : float, default=1.0
        Weight multiplier for the classification loss component.
    alpha_reg : float, default=1.0
        Weight multiplier for the regression loss component (after target normalization if enabled).
    dropout : float, default=0.1
        Dropout probability applied to the pooled backbone representation before each head.
    normalize_regression : bool, default=False
        If True, apply a normalization transform to the regression targets (train split statistics).
    reg_norm_method : str, default="standard"
        Normalization strategy if `normalize_regression` is True. One of:
        - "standard": (x - mean) / std
        - "robust"  : (x - median) / IQR
        - "minmax"  : Scaled to [-1, 1] via min-max
        Stored statistics are used for all splits and reversed at inference for regression outputs.   
    class_weighted : bool, default=False
        If True, applies balanced class weights to the classification loss based on training set frequencies.    
    model_name (derived) : str
        Internal name constructed as base_model + "-lora-dualhead".
    save_name (derived) : str
        Local directory name (final path: "./models/{save_name}").
    tokenizer_ (created) : transformers.PreTrainedTokenizer
        Tokenizer loaded from `base_model`.
    alpha_cls (stored) : float
        See above.
    alpha_reg (stored) : float
        See above.
    inverse_regression_target : callable or None
        Function to invert normalization for regression predictions (set if normalization enabled).

    Attributes (after fit)
    ----------------------
    model : torch.nn.Module
        The combined multitask model (backbone + dual heads + LoRA adapters).
    trainer_ : transformers.Trainer
        The custom Trainer handling multitask loss computation.
    train_dataset_ / test_dataset_ : datasets.Dataset
        Tokenized splits used for training/evaluation.
    training_args_ : transformers.TrainingArguments
        Final training arguments instance.
    training_results_ : dict
        Last evaluation metrics dict (e.g., after training).
    num_classes : int
        Number of distinct classification labels inferred from train split.
    reg_*_ (various)
        Stored normalization statistics (e.g., reg_mean_, reg_std_, reg_median_, reg_iqr_, reg_min_, reg_max_) depending on method.

    Methods (summary)
    -----------------
    fit(dataset)
        Train multitask model jointly on classification + regression targets.
    predict(dataset)
        Return dict with classification labels, probabilities, and regression predictions (denormalized if applicable).
    predict_cls(dataset)
        Return class indices only.
    predict_proba(dataset)
        Return softmax probabilities for classification head.
    predict_reg(dataset)
        Return regression predictions (denormalized if normalization applied).
    push(save_name=None, private=True)
        Push LoRA adapters and tokenizer (not full heads) to Hugging Face Hub.
    _normalize_regression_targets(...)
        Internal utility to fit/apply regression target normalization (called automatically when enabled).

    Loss Balancing Guidance
    -----------------------
    If MSE remains much smaller or larger than cross-entropy (e.g., due to unnormalized targets),
    enable normalization and/or adjust alpha_cls / alpha_reg to balance gradient magnitudes.
    Inspect logged losses to refine these weights.

    Metric Selection
    ----------------
    Set `metric_for_best_model` to any key produced by `_compute_metrics`. Classification keys
    (higher is better): "accuracy", "f1", "precision", "recall", "balanced accuracy".
    Regression keys (lower is better): "mse", "mae", "rmse"; higher is better: "r2".
    Optionally modify `training_args_.greater_is_better` after initialization for clarity.

    Example
    -------
        from datasets import Dataset, DatasetDict

        ds = DatasetDict({
            "train": Dataset.from_dict({
                "text": ["sample a", "sample b", "sample c"],
                "label_cls": [0, 1, 0],
                "label_reg": [2.5, 3.1, 2.9],
            }),
            "test": Dataset.from_dict({
                "text": ["example x", "example y"],
                "label_cls": [1, 0],
                "label_reg": [3.0, 2.7],
            })
        })

        model = AutoLoraForSeqDual(
            base_model="bert-base-uncased",
            metric_for_best_model="accuracy",
            alpha_cls=1.0,
            alpha_reg=1.0,
            normalize_regression=True,
            reg_norm_method="standard",
            target_modules="auto"
        )
        model.fit(ds)
        outputs = model.predict(ds["test"])
        print(outputs["cls"], outputs["reg"])

    Caveats
    -------
    - Only encoder-type models are assumed; adapting decoder-only architectures may require pooling changes.
    - Pushing to Hub saves only adapters + tokenizer; you must keep this class definition + head weights to reload the full multitask model.
    - Normalization statistics are not automatically serialized; persist them manually if needed for deployment.

    Raises
    ------
    ValueError
        If required dataset columns or splits are missing, if normalization method is invalid,
        or if model type / target modules cannot be resolved.

    """

    def __init__(
        self,
        metric_for_best_model: str = "accuracy",
        alpha_cls: float = 1.0,
        alpha_reg: float = 1.0,
        dropout: float = 0.1,
        normalize_regression: bool = False,
        reg_norm_method: str = "standard",
        class_weighted: bool = False,
        **kwargs,
    ):
        # Force a sensible default task type for encoder-only PEFT
        kwargs.setdefault("task_type", TaskType.FEATURE_EXTRACTION)
        super().__init__(metric_for_best_model=metric_for_best_model, **kwargs)
        self.alpha_cls = alpha_cls
        self.alpha_reg = alpha_reg
        self.dropout = dropout
        self.normalize_regression = normalize_regression
        self.reg_norm_method = reg_norm_method
        self.inverse_regression_target = None  # will be set if normalization is used
        self.class_weighted = class_weighted  
        self.class_weights_tensor: Optional[torch.Tensor] = None 

        # Override model/save names
        self.model_name = self.model_name + "-dualhead"
        self.save_name = self.model_name.split("/")[-1]

    def _set_destination_dir(self, save_name: Optional[str] = None):
        """
        Sets the destination directory for saving model checkpoints.
        
        Args: 
            save_name (Optional[str]): The name of the save directory. Defaults to self.save_name.

        Returns: 
            str: The full path to the destination directory.
        """
        if save_name is None:
            save_name = self.save_name
        return "./models/" + save_name

    def _get_classes(self, dataset: DatasetDict):
        """ 
        Determines the number of classes in the training dataset.

        Args: 
            dataset (DatasetDict): A Hugging Face DatasetDict containing the training split.

        Sets: 
            self.num_classes: The number of unique classes in the "label_cls" column of the training dataset.
        """
        self.num_classes = len(set(dataset["train"]["label_cls"]))

    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes the input examples and prepares classification and regression labels.

        Args: 
            examples (Dict[str, Any]): A dictionary containing the input examples with "text", "label_cls", and "label_reg" keys.

        Returns: 
            Dict[str, Any]: A dictionary containing tokenized inputs and processed labels for classification and regression.
        """

        tokenized = self.tokenizer_(examples["text"], truncation=True)
        tokenized["labels_cls"] = [int(y) for y in examples["label_cls"]]
        tokenized["labels_reg"] = [float(y) for y in examples["label_reg"]]
        return tokenized

    def _configure_lora(self):
        """
        Configures the LoRA (Low-Rank Adaptation) settings for the model.
       
        Performs: 
            - Dynamically infers target_modules if self.target_modules_ is set to "auto".
            - Initializes the LoRA configuration with parameters such as task type, rank, alpha, dropout, and target modules.

        Sets: 
            self.lora_config_: The configured LoraConfig instance.
        """
        # Allow dynamic target_modules inference if requested
        if self.target_modules_ == "auto":
            self._set_target_modules()

        self.lora_config_ = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # we're adapting the encoder backbone
            inference_mode=False,
            r=self.r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=self.target_modules_,
        )

    def _configure_training(self):
        """
        Configures the training arguments for the model.

        Sets up the Hugging Face TrainingArguments with parameters for training, evaluation, logging, and saving.

        Args: None

        Performs: 
            - Determines the output directory for saving model checkpoints. 
            - Configures training arguments such as learning rate, batch sizes, number of epochs, weight decay, and logging settings. 
            - Ensures compatibility with different versions of TrainingArguments by dynamically setting evaluation-related parameters: 
            - evaluation_strategy, eval_strategy, or evaluate_during_training, depending on the available signature.

        Results: 
            - Stores the configured TrainingArguments instance in self.training_args_. 
        """
        output_dir = self._set_destination_dir()

        # Build a version-compatible TrainingArguments
        kwargs = dict(
            output_dir=output_dir,
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
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=True,
        )

        sig_params = set(signature(TrainingArguments.__init__).parameters.keys())
        if "evaluation_strategy" in sig_params:
            kwargs["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in sig_params:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluate_during_training" in sig_params:
            kwargs["evaluate_during_training"] = True

        self.training_args_ = TrainingArguments(**kwargs)

    def _configure_trainer(self):
        """ 
        Configures the DualHeadTrainer for training and evaluation.

        Sets up the trainer with the model, datasets, data collator, and other training parameters.

        Args: None

        Performs: 
            - Initializes a DualHeadDataCollator for handling dual-head inputs. 
            - Prepares arguments for the DualHeadTrainer, including: 
            - alpha_cls and alpha_reg for task weighting. 
            - Model, training arguments, datasets, and metrics computation. 
            - Class weights for weighted loss computation. 
            - Dynamically checks if the DualHeadTrainer supports label_names in its constructor: 
            - If supported, passes label_names as ["labels_cls", "labels_reg"]. 
            - If not supported, sets label_names directly on the trainer instance.

        Results: 
            - Stores the configured DualHeadTrainer instance in self.trainer_. 
        """
        data_collator = DualHeadDataCollator(tokenizer=self.tokenizer_)

        trainer_kwargs = dict(
            alpha_cls=self.alpha_cls,
            alpha_reg=self.alpha_reg,
            model=self.model,
            args=self.training_args_,
            train_dataset=self.train_dataset_,
            eval_dataset=self.test_dataset_,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            class_weights=self.class_weights_tensor, 
        )

        # Pass label_names if this Trainer version supports it
        trainer_sig = set(signature(DualHeadTrainer.__init__).parameters.keys())
        if "label_names" in trainer_sig:
            trainer_kwargs["label_names"] = ["labels_cls", "labels_reg"]

        self.trainer_ = DualHeadTrainer(**trainer_kwargs)

        # If not supported via constructor, set directly on the Trainer instance
        if "label_names" not in trainer_sig:
            self.trainer_.label_names = ["labels_cls", "labels_reg"]

    def _normalize_regression_targets(
        self,
        dataset,
        column: str = "label_reg",
        method: str = "standard",
        add_inverse: bool = True,
        eps: float = 1e-8,
    ):
        """
        Normalize the regression targets in-place so the regression loss (MSE) is on a scale
        comparable to the classification loss (cross-entropy).

        Parameters
        ----------
        dataset : datasets.DatasetDict
            Must contain at least a 'train' split with the regression target column.
        column : str
            Name of the regression target column to normalize.
        method : str
            One of:
            - 'standard' : (x - mean) / std
            - 'robust'   : (x - median) / IQR   (IQR = q75 - q25; falls back to std-like scaling if IQR≈0)
            - 'minmax'   : (x - min) / (max - min) scaled into [-1, 1] (centered range)
        add_inverse : bool
            If True, defines self.inverse_regression_target(y_norm) to map normalized values back
            to the original scale.
        eps : float
            Small constant to avoid division by zero.

        Returns
        -------
        dataset : datasets.DatasetDict
            The same object with the specified column normalized in every split present.

        Notes
        -----
        - Fits statistics ONLY on the 'train' split; applies the same transform to all other splits.
        - Stores fitted parameters as attributes on `self` (e.g., self.reg_mean_, self.reg_std_, etc.)
        - Safe to call multiple times: subsequent calls will overwrite previous normalization.
        - Call this BEFORE tokenization / mapping so the transformed values are what the model sees.
        """

        if "train" not in dataset:
            raise ValueError("DatasetDict must contain a 'train' split to fit normalization.")

        train_vals = np.asarray(dataset["train"][column], dtype=float)

        if method == "standard":
            mean = float(train_vals.mean())
            std = float(train_vals.std())
            if std < eps:
                std = 1.0
            self.reg_mean_ = mean
            self.reg_std_ = std

            def _forward(x):
                return (x - mean) / std

            if add_inverse:
                def inverse_fn(y):
                    return y * std + mean
                self.inverse_regression_target = inverse_fn

        elif method == "robust":
            q25, q75 = np.percentile(train_vals, [25, 75])
            median = float(np.median(train_vals))
            iqr = float(q75 - q25)
            if iqr < eps:
                # Fallback to std scaling if IQR is degenerate
                iqr = float(train_vals.std()) or 1.0
            self.reg_median_ = median
            self.reg_iqr_ = iqr

            def _forward(x):
                return (x - median) / iqr

            if add_inverse:
                def inverse_fn(y):
                    return y * iqr + median
                self.inverse_regression_target = inverse_fn

        elif method == "minmax":
            vmin = float(train_vals.min())
            vmax = float(train_vals.max())
            span = vmax - vmin
            if span < eps:
                span = 1.0
            # Scale to [0,1] then shift to [-1,1] for zero-centering
            self.reg_min_ = vmin
            self.reg_max_ = vmax

            def _forward(x):
                return 2.0 * ((x - vmin) / span) - 1.0

            if add_inverse:
                def inverse_fn(y):
                    return ((y + 1.0) / 2.0) * span + vmin
                self.inverse_regression_target = inverse_fn
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from ['standard', 'robust', 'minmax'].")

        self.reg_norm_method_ = method
        self.reg_norm_column_ = column

        # Apply transform to every split present
        def _map_fn(example):
            val = example[column]
            # Support scalar or list (just in case)
            if isinstance(val, list):
                example[column] = [_forward(float(v)) for v in val]
            else:
                example[column] = _forward(float(val))
            return example

        for split_name in dataset.keys():
            dataset[split_name] = dataset[split_name].map(_map_fn)

        return dataset

    def fit(self, dataset: DatasetDict):
        """ 
        Fine-tunes the dual-head model on the provided dataset.

        Args: 
            dataset (DatasetDict): A Hugging Face DatasetDict containing 'train' and 'test' splits. 
            Each split must include the columns 'text', 'label_cls', and 'label_reg'.

        Raises: 
            ValueError: If the dataset is not a DatasetDict or is missing required splits or columns.

        Performs: 
            - Validation of the dataset structure and columns. 
            - Class weight computation for balanced training if class_weighted is enabled. 
            - Normalization of regression targets if normalize_regression is enabled. 
            - Construction of a PEFT-wrapped backbone and dual-head model. 
            - Gradient checkpointing and tokenizer adjustments. 
            - Dataset mapping and trainer configuration. 
            - Model training and evaluation.

        Results: 
            - Stores training results in self.training_results_. 
            - Prints evaluation metrics in a tabular format. 
        """
        # Validate dataset
        if not isinstance(dataset, DatasetDict) or "train" not in dataset or "test" not in dataset:
            raise ValueError("Dataset must be a Huggingface DatasetDict with 'train' and 'test' splits.")
        required_cols = {"text", "label_cls", "label_reg"}
        for split in ("train", "test"):
            missing = required_cols - set(dataset[split].column_names)
            if missing:
                raise ValueError(f"Missing columns in split '{split}': {missing}")

        # Determine number of classes
        self._get_classes(dataset)

        # Compute balanced class weights if requested
        if self.class_weighted:
            train_labels = list(dataset["train"]["label_cls"])
            counts = np.bincount(train_labels, minlength=self.num_classes).astype(float)
            N = counts.sum()
            K = self.num_classes
            weights = N / (K * counts)  # balanced scheme
            # Normalize so mean weight = 1 to keep CE scale stable
            weights = weights / (weights.mean() + 1e-12)
            self.class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            self.class_weights_tensor = None

        # Normalize regression targets if requested
        if self.normalize_regression:
            dataset = self._normalize_regression_targets(dataset, method=self.reg_norm_method, add_inverse=True)

        # Build backbone and apply LoRA to the backbone only
        backbone = AutoModel.from_pretrained(self.base_model)
        backbone = get_peft_model(backbone, self.lora_config_)

        # Build dual-head model around the PEFT-wrapped backbone
        self.model = MultiTaskSequenceModel(backbone=backbone, num_labels=self.num_classes, dropout=self.dropout)
        self.model.train()

        # Gradient checkpointing and cache settings (on the backbone if available)
        if self.gradient_checkpointing:
            try:
                if hasattr(self.model.backbone, "gradient_checkpointing_enable"):
                    self.model.backbone.gradient_checkpointing_enable()
                config = getattr(self.model.backbone, "config", None)
                if config is not None and hasattr(config, "use_cache"):
                    config.use_cache = False
            except Exception:
                pass

        # Ensure tokenizer has a padding token and resize embeddings on the backbone
        if self.tokenizer_.pad_token is None:
            self.tokenizer_.add_special_tokens({"pad_token": "[PAD]"})
            try:
                if hasattr(self.model.backbone, "resize_token_embeddings"):
                    self.model.backbone.resize_token_embeddings(len(self.tokenizer_))
            except Exception:
                pass

        # Map datasets
        self.train_dataset_ = self._map_dataset(dataset["train"])
        self.test_dataset_ = self._map_dataset(dataset["test"])

        # Configure trainer and train
        self._configure_trainer()
        self.trainer_.train()

        # Evaluate
        self.training_results_ = self.trainer_.evaluate()
        table = tabulate(self.training_results_.items(), headers=["Metric", "Value"], tablefmt="simple_outline")
        print(table)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Computes evaluation metrics for both classification and regression tasks.

        Args: eval_pred: An object containing predictions and labels. 
            - eval_pred.predictions: Model predictions, which can be a tuple (logits_cls, preds_reg) or a single array. 
            - eval_pred.label_ids: Ground truth labels, which can be a tuple (labels_cls, labels_reg) or a single array.

        Returns: Dict[str, float]: A dictionary containing the computed metrics. Includes: 
            - Classification metrics: "accuracy", "balanced accuracy", "f1", "precision", "recall". 
            - Regression metrics: "mse", "mae", "r2", "rmse". 
        """
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        # Unpack predictions
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            logits_cls, preds_reg = preds
        else:
            logits_cls, preds_reg = preds, None

        # Unpack labels (HF returns tuple/list in same order as label_names)
        if isinstance(labels, (list, tuple)) and len(labels) == 2:
            labels_cls, labels_reg = labels
        else:
            labels_cls, labels_reg = labels, None

        metrics: Dict[str, float] = {}

        # Classification metrics
        if logits_cls is not None and labels_cls is not None:
            y_pred_cls = np.argmax(logits_cls, axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_cls, y_pred_cls, average="weighted")
            acc = accuracy_score(labels_cls, y_pred_cls)
            bacc = balanced_accuracy_score(labels_cls, y_pred_cls)
            metrics.update(
                {
                    "accuracy": acc,
                    "balanced accuracy": bacc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

        # Regression metrics
        if preds_reg is not None and labels_reg is not None:
            preds_reg_flat = preds_reg.reshape(-1)
            labels_reg_flat = labels_reg.reshape(-1)
            mse = mean_squared_error(labels_reg_flat, preds_reg_flat)
            mae = mean_absolute_error(labels_reg_flat, preds_reg_flat)
            r2 = r2_score(labels_reg_flat, preds_reg_flat)
            if root_mean_squared_error is not None:
                rmse = root_mean_squared_error(labels_reg_flat, preds_reg_flat)
            else:
                rmse = float(np.sqrt(mse))
            metrics.update({"mse": mse, "mae": mae, "r2": r2, "rmse": rmse})

        return metrics

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """ 
        Predicts class probabilities for the given dataset.

        Args: 
            dataset (Dataset): A Hugging Face Dataset to predict on.

        Returns: 
            np.ndarray: A 2D array of class probabilities for each sample in the dataset. 
        """
        dataset = self._map_dataset(dataset)
        predictions = self.trainer_.predict(dataset)
        logits_cls, _ = predictions.predictions  # (cls_logits, reg_preds)
        # numerically stable softmax
        logits_max = np.max(logits_cls, axis=1, keepdims=True)
        exp_shifted = np.exp(logits_cls - logits_max)
        proba = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        
        return proba

    def predict_cls(self, dataset: Dataset) -> np.ndarray:
        """ 
        Predicts class labels for the given dataset.

        Args: dataset (Dataset): A Hugging Face Dataset to predict on.

        Returns: np.ndarray: A 1D array of predicted class labels for each sample in the dataset. 
        """
        proba = self.predict_proba(dataset)
        
        return np.argmax(proba, axis=1)

    def predict_reg(self, dataset: Dataset) -> np.ndarray:
        """ Predicts regression values for the given dataset.

        Args: dataset (Dataset): A Hugging Face Dataset to predict on.

        Returns: np.ndarray: A 1D array of predicted regression values for each sample in the dataset. 
        """
        dataset = self._map_dataset(dataset)
        predictions = self.trainer_.predict(dataset)
        _, preds_reg = predictions.predictions
        preds_reg = preds_reg.reshape(-1)

        if self.inverse_regression_target is not None:
            preds_reg = self.inverse_regression_target(preds_reg)

        return preds_reg

    def predict(self, dataset: Dataset) -> Dict[str, np.ndarray]:
        """ 
        Predicts both class labels and regression values for the given dataset.

        Args: dataset (Dataset): 
            A Hugging Face Dataset to predict on.

        Returns: 
            Dict[str, np.ndarray]: A dictionary containing: 
            - "cls": Predicted class labels (1D array). 
            - "proba": Class probabilities (2D array). 
            - "reg": Predicted regression values (1D array). 
        """
        dataset = self._map_dataset(dataset)
        predictions = self.trainer_.predict(dataset)
        logits_cls, preds_reg = predictions.predictions

        preds_reg = preds_reg.reshape(-1)
        if self.inverse_regression_target is not None:
            preds_reg = self.inverse_regression_target(preds_reg)

        logits_max = np.max(logits_cls, axis=1, keepdims=True)
        proba = np.exp(logits_cls - logits_max)
        proba /= np.sum(proba, axis=1, keepdims=True)
        y_cls = np.argmax(proba, axis=1)
        return {"cls": y_cls, "proba": proba, "reg": preds_reg}

