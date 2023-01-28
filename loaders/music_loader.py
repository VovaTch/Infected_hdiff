import os
from typing import List
import tempfile

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import tqdm


class MP3SliceDataset(Dataset):
    """
    This class creates a dataset of slices of MP3 files in a folder as a Mel Spectrogram form.
    """

    def __init__(self, 
                 sample_rate: int=44100,
                 audio_dir: str="data/music_samples/", 
                 slice_time: float=5.0,
                 n_fft: int=1024,
                 hop_length: int=512,
                 n_melts_per_second: int=64,
                 device: str="cpu",
                 **kwargs):
        
        # Initialize the object variables
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir
        self.slice_time = slice_time
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_melts_per_second = n_melts_per_second
        self.num_samples = int(n_melts_per_second * slice_time)
        
        # Important: total samples in a slice
        self.total_samples_per_slice = int(self.sample_rate * self.slice_time)
        
        # Create a file list
        file_list = []
        for file in os.listdir(self.audio_dir):
            if file.endswith("mp3"):
                file_list.append(os.path.join(self.audio_dir, file))
        self.processed_slice_data = self._create_music_slices(file_list).squeeze(0)
        
    def _create_music_slices(self, file_list: List[str]):
        """
        Private method to create music slices of 5 seconds from mp3 files.
        """
        
        total_slices = torch.zeros((1, 0, self.total_samples_per_slice))
        
        for file in tqdm.tqdm(file_list, desc='Loading music slices...'):
            long_data, sr = torchaudio.load(file, format="mp3")
            long_data = self._resample_if_necessary(long_data, sr)
            long_data = self._mix_down_if_necessary(long_data)
            long_data = self._right_pad_if_necessary(long_data)
            slices = long_data.view((1, -1, self.total_samples_per_slice))
            total_slices = torch.cat((total_slices, slices), dim=1)
            
        return total_slices
    
            
    def _resample_if_necessary(self, signal: torch.Tensor, sr: int):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal
    
    
    def _mix_down_if_necessary(self, signal: torch.Tensor):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
            
            
    def _right_pad_if_necessary(self, signal: torch.Tensor):
        length_signal = signal.shape[1]
        if length_signal % self.total_samples_per_slice != 0:
            num_missing_samples = self.total_samples_per_slice - length_signal % self.total_samples_per_slice
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal
        
    def __getitem__(self, idx):
        return {'music slice': self.processed_slice_data[idx].unsqueeze(0)}
    
    def __len__(self):
        return self.processed_slice_data.shape[0]
    
def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()
    
if __name__ == "__main__":
    
    dataset = MP3SliceDataset()
    
    sliced_piece = dataset[0]
    sliced_piece = sliced_piece.unsqueeze(0)
    print(sliced_piece.size())
    
    with tempfile.TemporaryDirectory() as tempdir:
        path = f"save_example_default.mp3"
        torchaudio.save(path, sliced_piece, 44100, format="mp3")
        inspect_file(path)