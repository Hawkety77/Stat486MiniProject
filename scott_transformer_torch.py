from huggingface_hub import login
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import ViTImageProcessor
from datasets import load_dataset
from datasets import load_metric
from transformers import TrainingArguments
import numpy as np
import torch

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ToPILImage
)

## Define constants
MODEL_CHECKPOINT = "MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection" #"prashanth0205/vit_spectrogram"
BATCH_SIZE = 32 #16
HUGGINGFACE_KEY = ''

## Load Vision Transformer image processor
image_processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)

# option 1: local/remote files (supporting the following formats: tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", split='train',
                       data_dir="/home/scottdb1/Stat486MiniProject/Data/images_original")
#"/home/scottbrown/byu/stat486/projects/Stat486MiniProject/Data/images_original")

## Create normalizer
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

## Define data transforms
val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

## Preprocessor
def preprocess(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
metric = load_metric("accuracy")

## Define train args
model_name = MODEL_CHECKPOINT.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=0.01, #5e-5
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=150,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False, # don't auto-push back to Hugging Face
)

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    met = metric.compute(predictions=predictions, references=eval_pred.label_ids)
    print(met) # important to print here!
    return met

## Collator function strings batches together
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

splits = dataset.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

## Define model
model = AutoModelForImageClassification.from_pretrained(MODEL_CHECKPOINT, from_tf=False, ignore_mismatched_sizes = True,
                                                        num_labels=len(labels))

login(HUGGINGFACE_KEY) # authenticate HF to load Trainer

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

## Train!
train_results = trainer.train()
print(train_results)

# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()