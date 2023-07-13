import os
from typing import List
import tempfile
import random

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import tqdm
import pickle


class MP3SliceDataset(Dataset):
    """
    This class creates a dataset of slices of MP3 files in a folder as a Mel Spectrogram form.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        audio_dir: str = "data/music_samples/",
        slice_time: float = 0.7430386,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_melts_per_second: int = 64,
        device: str = "cpu",
        preload: bool = True,
        preload_data_file_path: str = "data/music_samples/000-datatensor.pt",
        preload_metadata_file_path: str = "data/music_samples/000-metadata.pkl",
        **kwargs,
    ):
        # Initialize the object variables
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir
        self.slice_time = slice_time
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_melts_per_second = n_melts_per_second
        self.num_samples = int(n_melts_per_second * slice_time)
        self.preload = preload
        self.device = device

        # Important: total samples in a slice
        self.total_samples_per_slice = int(self.sample_rate * self.slice_time)

        # Create a file list
        self.processed_slice_data = None
        self.file_list = []
        self.metadata = []
        for file in os.listdir(self.audio_dir):
            if file.endswith("mp3"):
                self.file_list.append(os.path.join(self.audio_dir, file))
        if self.preload:
            # self.processed_slice_data = self._create_music_slices(self.file_list).squeeze(0).to(device)

            # Load pt file if exists
            if os.path.isfile(preload_data_file_path):
                print(f"Loading file {preload_data_file_path}...")
                data_dict = torch.load(preload_data_file_path)
                self.processed_slice_data = data_dict["slices"]
                print(f"Music file {preload_data_file_path} is loaded.")

            # Save pt file if not
            else:
                (
                    self.processed_slice_data,
                    self.metadata,
                    self.time_stamps,
                ) = self._create_music_slices(self.file_list)
                self.processed_slice_data = self.processed_slice_data.squeeze(0).to(
                    device
                )
                data_dict = {"slices": self.processed_slice_data}
                torch.save(data_dict, preload_data_file_path)
                print(f"Saved music file at {preload_data_file_path}")

            # Load pickle file if exists
            if os.path.isfile(preload_metadata_file_path):
                print(f"Loading file {preload_metadata_file_path}...")
                with open(preload_metadata_file_path, "rb") as f:
                    data_dict = pickle.load(f)
                    self.metadata, self.time_stamps = (
                        data_dict["track names"],
                        data_dict["time stamps"],
                    )
                print(f"Music file {preload_metadata_file_path} is loaded.")

            # Save pickle file if not
            else:
                if self.metadata is None:
                    _, self.metadata, self.time_stamps = (
                        self._create_music_slices(self.file_list).squeeze(0).to(device)
                    )
                with open(preload_metadata_file_path, "wb") as f:
                    data_dict = {
                        "track names": self.metadata,
                        "time stamps": self.time_stamps,
                    }
                    pickle.dump(data_dict, f)
                print(f"Saved metadata file at {preload_metadata_file_path}")

    def _create_music_slices(self, file_list: List[str]):
        """
        Private method to create music slices of 5 seconds from mp3 files.
        """

        total_slices = torch.zeros((1, 0, self.total_samples_per_slice))
        track_name_list = []
        time_stamps_total = []

        for file in tqdm.tqdm(file_list, desc="Loading music slices..."):
            slices, track_name, time_stamps = self._load_slices_from_track(file)
            track_name_list += track_name
            total_slices = torch.cat((total_slices, slices), dim=1)
            time_stamps_total += time_stamps

        return total_slices, track_name_list, time_stamps_total

    def _load_slices_from_track(self, file: str):
        long_data, sr = torchaudio.load(file, format="mp3")
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)
        long_data = self._right_pad_if_necessary(long_data)
        slices = long_data.view((1, -1, self.total_samples_per_slice))
        track_name_list = [[file] for _ in range(slices.shape[1])]
        time_stamp_list = [self.slice_time * idx for idx in range(slices.shape[1])]

        return slices, track_name_list, time_stamp_list

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
            num_missing_samples = (
                self.total_samples_per_slice
                - length_signal % self.total_samples_per_slice
            )
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal

    def __getitem__(self, idx):
        if self.preload:
            slice = self.processed_slice_data[idx]
            track_name = self.metadata[idx]
            time_stamp = self.time_stamps[idx]
        else:
            file_choice = random.choice(self.file_list)
            slices, track_name_repeated, time_stamp = self._load_slices_from_track(
                file_choice
            ).squeeze(0)
            slice_choice = random.choice(list(range(0, slices.shape[0])))
            slice = slices[slice_choice, ...]
            track_name = track_name_repeated[0]

        return {
            "music slice": slice.unsqueeze(0).to(self.device),
            "track name": track_name[0],
            "time stamp": time_stamp,
        }

    def __len__(self):
        return self.processed_slice_data.shape[0] if self.preload else 100


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
        path = "save_example_default.mp3"
        torchaudio.save(path, sliced_piece, 44100, format="mp3")
        inspect_file(path)
