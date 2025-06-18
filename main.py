import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from model.model_CNN_LSTM import CNNLSTM
from utilities.data import Firmness_dataset
from utilities.utils import Arguments
from transformers import (
    AutoImageProcessor, AutoConfig, AutoModelForVideoClassification,
    EarlyStoppingCallback, TrainingArguments, Trainer, set_seed, HfArgumentParser
)
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import plotly.express as px


def get_optimizer_and_scheduler(model, optim_name, learning_rate, weight_decay, num_training_steps):
    if optim_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim_name == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optim_name == "RMS":
        optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    return optimizer, scheduler

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    r2 = r2_score(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"r2": r2, "rmse": rmse, "mae": mae}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

CNN_transforms = None
image_processor = None
parser = HfArgumentParser((Arguments,))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
else:
    args = parser.parse_args_into_dataclasses()

set_seed(args.seed)
os.environ['WANDB_DISABLED'] = 'true'

if args.video_model == "CNNLSTM":
    if args.CNN_LSTM_loss_fnc == "SmoothL1":
        loss_func = nn.SmoothL1Loss()
    elif args.CNN_LSTM_loss_fnc == "MSEloss":
        loss_func = nn.MSELoss()
    elif args.CNN_LSTM_loss_fnc == "Huberloss":
        loss_func = nn.HuberLoss()

    model = CNNLSTM(
        num_labels=1,
        hidden_size=args.LSTM_hidden_size,
        num_lstm_layers=args.LSTM_layers,
        loss_func=loss_func,
        pretrained=args.pre_trained_weights
    )

    CNN_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    image_processor = AutoImageProcessor.from_pretrained(args.image_processor)
    if args.pre_trained_weights:
        model = AutoModelForVideoClassification.from_pretrained(
            args.video_model, num_labels=1, ignore_mismatched_sizes=True
        )
    else:
        config = AutoConfig.from_pretrained(args.video_model, num_labels=1)
        model = AutoModelForVideoClassification.from_config(config=config)

data_df = pd.read_csv(args.data_csv)
train_df = data_df[data_df["split"] == "train"].reset_index(drop=True)
valid_df = data_df[data_df["split"] == "val"].reset_index(drop=True)
test_df = valid_df.copy()

train_dataset = Firmness_dataset(
    data_root=args.data_root, image_processor=image_processor, df=train_df,
    sample_duration=args.sample_duration, transform=CNN_transforms, video_model=args.video_model
)

valid_dataset = Firmness_dataset(
    data_root=args.data_root, image_processor=image_processor, df=valid_df,
    sample_duration=args.sample_duration, transform=CNN_transforms, video_model=args.video_model
)

test_dataset = Firmness_dataset(
    data_root=args.data_root, image_processor=image_processor, df=test_df,
    sample_duration=args.sample_duration, transform=CNN_transforms, video_model=args.video_model
)

print("Train size =", len(train_dataset))
print("Valid size =", len(valid_dataset))
print("Test size =", len(test_dataset))

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

train_args = TrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="eval_r2",
    logging_dir='logs',
    max_grad_norm=1.0
)

num_training_steps = len(train_dataset) // train_args.per_device_train_batch_size * train_args.num_train_epochs

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    optimizers=get_optimizer_and_scheduler(
        model, optim_name=args.optim_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_training_steps=num_training_steps
    ),
)

trainer.train(resume_from_checkpoint=args.resume_chk)

output = trainer.predict(test_dataset)
predictions = output.predictions.squeeze()
labels = output.label_ids

results_df = pd.DataFrame({
    "Ground_Truth": labels,
    "Predictions": predictions
})
results_df = results_df.sort_values(by="Ground_Truth")
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

fig = px.scatter(results_df, x="Ground_Truth", y="Predictions", trendline="ols")
fig.write_image(os.path.join(output_dir, "plot_1.png"))

r2 = r2_score(labels, predictions)
rmse = root_mean_squared_error(labels, predictions)
mae = mean_absolute_error(labels, predictions)

with open(os.path.join(output_dir, "results.txt"), "w") as f:
    f.write(f"R squared: {r2:.2f}%\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
