import os
import cv2
import torch
from torch.utils.data import Dataset

class Firmness_dataset(Dataset):
    
    def __init__(self, data_root, df, sample_duration,image_processor=None, transform=None,video_model=None):
        """
        Dataset for video frames and firmness labels without an external image processor.

        Args:
            data_root (str): Root directory for video data.
            df (pd.DataFrame): DataFrame containing metadata (path, labels, num_frames).
            sample_duration (int): Number of frames to sample per video.
            transform (callable, optional): Transformations to apply to each frame.
        """
        self.data_root = data_root
        self.df = df
        self.sample_duration = sample_duration
        self.transform = transform
        self.video_model = video_model
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        readings_path = self.data_root + self.df["path"][idx]
        firmness_label = float(self.df["average_firmness"][idx])
        num_frames = self.df["num_frames"][idx]

        frames = os.listdir(readings_path)
        frames.sort()

        step = max(1, (num_frames - 1) / (self.sample_duration - 1))
        frames_index = [round(step * i) for i in range(self.sample_duration)]

        video = []
        for i in frames_index:
            frame_path = os.path.join(readings_path, frames[i])
            try:
                frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            except:
                print(frame_path)
            if self.transform:
                frame = self.transform(frame)
            video.append(frame)
            
        if self.video_model == "CNNLSTM":
            video_tensor = torch.stack(video).squeeze()
            inputs = {"pixel_values": video_tensor, "labels": firmness_label}

        else:
            inputs  = self.image_processor(video, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].squeeze()
            inputs["labels"] = firmness_label

        inputs["path"] = self.df["path"][idx]
        inputs["num_frames"] = num_frames

        return inputs