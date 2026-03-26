#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Train a HuggingFace audio classification model for orca call detection.

This script uses the Wav2Vec2 model architecture fine-tuned on orca call audio.
The trained model can be pushed to HuggingFace Hub or saved locally.

Usage:
    python train_huggingface_model.py --output_dir ./model --push_to_hub
"""

import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import EvalPrediction
import evaluate

# Get repository root (ModelTraining directory).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Label mapping.
LABEL2ID = {"resident": 0, "transient": 1, "humpback": 2, "other": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Load metrics once at module scope.
ACCURACY_METRIC = evaluate.load("accuracy")
PRECISION_METRIC = evaluate.load("precision")
RECALL_METRIC = evaluate.load("recall")
F1_METRIC = evaluate.load("f1")


def load_audio_dataset(data_dir: Path) -> DatasetDict:
    """
    Load audio files from the output/wav directory structure.
    
    Args:
        data_dir: Path to ModelTraining/output/wav directory
        
    Returns:
        DatasetDict with train and test splits
        
    Raises:
        ValueError: If no audio files are found or dataset is too small for training
    """
    audio_files = []
    labels = []
    
    # Iterate through each category directory.
    for category in ["resident", "transient", "humpback", "other"]:
        category_dir = data_dir / category
        if not category_dir.exists():
            print(f"Warning: {category_dir} does not exist")
            continue
            
        # Get all wav files in this category.
        wav_files = list(category_dir.glob("**/*.wav"))
        print(f"Found {len(wav_files)} files for {category}")
        
        for wav_file in wav_files:
            audio_files.append(str(wav_file))
            labels.append(LABEL2ID[category])
    
    # Validate that dataset is not empty.
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_dir}. Please ensure wav files exist in category subdirectories.")
    
    # Check label distribution for stratification
    label_counts = Counter(labels)
    min_samples_per_label = min(label_counts.values())
    
    print(f"Total samples: {len(audio_files)}")
    print(f"Label distribution: {dict(label_counts)}")
    
    # Create dataset.
    dataset = Dataset.from_dict({
        "audio": audio_files,
        "label": labels
    })
    
    # Cast audio column to Audio type.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split into train/test.
    # Use stratification only if each label has at least 2 samples and total > 1.
    if len(audio_files) < 2:
        raise ValueError(f"Dataset too small for train/test split: only {len(audio_files)} sample(s) found. Need at least 2 samples.")
    
    # Check if stratification is possible:
    # - Need at least 2 unique labels
    # - Each label must have at least 2 samples
    num_unique_labels = len(label_counts)
    can_stratify = num_unique_labels > 1 and min_samples_per_label >= 2
    
    if can_stratify:
        # Stratified split (maintains class distribution).
        print("Using stratified train/test split (80/20)")
        dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    else:
        # Cannot use stratification - use random split.
        if num_unique_labels == 1:
            print(f"Warning: Dataset has only 1 unique label. Using non-stratified split.")
        elif min_samples_per_label < 2:
            print(f"Warning: Some labels have fewer than 2 samples (min={min_samples_per_label}). Using non-stratified split.")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset


def preprocess_function(examples: dict, feature_extractor: Wav2Vec2FeatureExtractor, max_duration: float = 3.0) -> dict:
    """
    Preprocess audio files for the model.
    
    Args:
        examples: Batch of examples with audio data
        feature_extractor: Wav2Vec2FeatureExtractor instance
        max_duration: Maximum audio duration in seconds
        
    Returns:
        Processed inputs for the model (as NumPy arrays for serialization)
    """
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    # Pad or truncate to max_duration. Wav2Vec2 expects fixed-length inputs.
    target_length = int(max_duration * 16000)  # 16kHz sample rate.
    processed_audio = []
    for audio in audio_arrays:
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        processed_audio.append(audio)
    
    # Return NumPy arrays instead of PyTorch tensors for dataset caching/serialization.
    # The Trainer's data collator will convert to tensors during batching.
    inputs = feature_extractor(
        processed_audio,
        sampling_rate=16000,
        padding=True,
        max_length=target_length,
        truncation=True,
    )
    
    # Ensure input_values is a NumPy array (feature_extractor returns this by default).
    # Convert to list of arrays for proper serialization in datasets library.
    inputs["labels"] = examples["label"]
    
    return inputs


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    
    accuracy = ACCURACY_METRIC.compute(predictions=predictions, references=eval_pred.label_ids)
    precision = PRECISION_METRIC.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")
    recall = RECALL_METRIC.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")
    f1 = F1_METRIC.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train HuggingFace audio classification model for orca calls"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output/wav",
        help="Path to wav files directory (default: output/wav)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/huggingface",
        help="Directory to save the trained model (default: model/huggingface)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="Base model to fine-tune (default: facebook/wav2vec2-base)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="orca-call-detector",
        help="HuggingFace Hub model ID (default: orca-call-detector)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a specific checkpoint to resume training from",
    )
    args = parser.parse_args()
    
    # Set up paths.
    data_dir = REPO_ROOT / args.data_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {data_dir}...")
    dataset = load_audio_dataset(data_dir)
    print(f"Dataset: {dataset}")
    
    # Load feature extractor and model.
    print(f"Loading feature extractor and model: {args.model_name}")
    
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    except Exception as e:
        error_msg = f"Error loading feature extractor from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e
    
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
    except Exception as e:
        error_msg = f"Error loading model from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e
    
    # Preprocess dataset.
    print("Preprocessing dataset...")
    dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        remove_columns=["audio"],
    )
    
    # Training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )
    
    # Create trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    # Resume from checkpoint if provided.
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        # Train.
        print("Starting training...")
        trainer.train()
    
    # Evaluate.
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")
    
    # Save model and feature extractor.
    print(f"Saving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))
    
    print("Training complete!")
    
    if args.push_to_hub:
        print(f"Model pushed to HuggingFace Hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
