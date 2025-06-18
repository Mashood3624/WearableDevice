import torch.nn as nn
from torchvision.models import mobilenet_v2
from safetensors.torch import load_file

class CNNLSTM(nn.Module):
    def __init__(self, num_labels, hidden_size=256, num_lstm_layers=2,
                 loss_func= nn.SmoothL1Loss(), pretrained=True):
        """
        CNN-LSTM model for fruit firmness estimation

        Args:
            num_labels (int): Number of output labels (1 for regression).
            hidden_size (int): Number of hidden units in the LSTM.
            num_lstm_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate for regularization.
            freeze_resnet (bool): Whether to freeze ResNet layers during training.
        """
        super(CNNLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.pretrained = pretrained
        self.loss_func = loss_func
        self.num_labels = num_labels

        backbone = mobilenet_v2()
        self.feature_extractor = nn.Sequential(*list(backbone.features))
        if self.pretrained:
            loaded_state_dict = load_file("./weights/cnn_backbone.safetensors") # pretrained tactile weights for encoder
            self.feature_extractor.load_state_dict(loaded_state_dict)
        self.feature_dim = 1280 
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, 
                            num_layers=self.num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, pixel_values, labels=None):
        batch_size, time_steps, c, h, w = pixel_values.shape
        x = pixel_values.view(batch_size * time_steps, c, h, w) 
        features = self.feature_extractor(x)  
        features = features.mean([2, 3]) 
        features = features.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(features)
        last_lstm_out = lstm_out[:, -1, :] 
        
        logits = self.fc(last_lstm_out)

        if labels is not None:
            loss_fct = self.loss_func
            loss = loss_fct(logits.view(-1), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits, 'lstm_out':lstm_out}