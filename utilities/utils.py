from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    output_dir: str = field(
        metadata={"help": "Path to save the outputs"},
    )
    
    video_model: str = field(
        metadata={"help": "Model name or path of vision encoder"},
    )

    data_root: str = field(
        metadata={"help": "Path to dataset"},
    )
    
    data_csv: str = field(
        metadata={"help": "Path for train dataset CSV file"},
    )
    
    sample_duration: int = field(
        metadata={"help": "Number of smaples per video"},
    )
    
    seed: Optional[str] = field(
        default=1234, metadata={"help": "Batch size at training phase"}
    )

    image_processor: Optional[str] = field(
        default="facebook/timesformer-base-finetuned-k400", metadata={"help": "Model name or path of vision encoder"},
    )

    optim_name: Optional[str] = field(
        default="AdamW", metadata={"help": "Argument to load pre_trained weights of vision model"}
    )

    CNN_LSTM_loss_fnc: Optional[str] = field(
        default="SmoothL1", metadata={"help": "Argument to load pre_trained weights of vision model"}
    )

    LSTM_hidden_size: Optional[int] = field(
        default=128, metadata={"help": "Argument to load pre_trained weights of vision model"}
    )

    LSTM_layers: Optional[int] = field(
        default=1, metadata={"help": "Argument to load pre_trained weights of vision model"}
    )
    
    pre_trained_weights: Optional[bool] = field(
        default=True, metadata={"help": "Argument to load pre_trained weights of vision model"}
    )
    
    freeze_weights: Optional[bool] = field(
        default=False, metadata={"help": "Batch size at training phase"}
    )

    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size at training phase"}
    )
    
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size at evaluation phase"}
    )
    
    num_train_epochs: Optional[int] = field(
        default=50, metadata={"help": "Total number of epochs"}
    )
    
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Weight decay rate for training the model"}
    )
    
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "Learning rate for training the model"}
    )
    
    resume_chk: Optional[bool] = field(
        default=False, metadata={"help": "Resume training from last checkpoint"}
    )