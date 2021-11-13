import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa
from utilities import time_stamp_to_z
import numpy as np
MAP={"speech":0," music":1,"speech_music":2}

class AudioDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        col_names=["filename","start","end","type","class"]
        self.num_samples=num_samples
        self.target_sample_rate=target_sample_rate
        self.device = device
        self.annotations = pd.read_excel(annotations_file,index_col="filename")
        self.audio_dir = audio_dir
        self.audio_sample_list=self._get_audio_sample_path(audio_dir)
        self.transformation = transformation.to(self.device)
        

    def __len__(self):
        return len(self.audio_sample_list)

    def __getitem__(self, index):
        audio_sample_path = self.audio_sample_list[index]
        label = self._get_audio_sample_label(audio_sample_path[25:])
        signal, sr = librosa.load(audio_sample_path,self.target_sample_rate)
        #signal = self._right_pad_if_necessary(signal)
        
        if(len(signal)<self.num_samples):
            signal=np.pad(signal,(0,self.num_samples-len(signal)),'constant',constant_values=0)
            
        #signal = librosa.feature.melspectrogram(signal,sr=sr,n_fft=1024,hop_length=512,win_length=1024)
        signal=np.abs(librosa.stft(signal, n_fft = 1024, hop_length = 512, win_length = 1024, window='hann')) 
        signal=librosa.power_to_db(signal**2,ref=np.max)
        signal=librosa.feature.melspectrogram(S=signal)
        signal=(signal-signal.mean())/signal.std()
        
        return signal, label

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[0]
        print(length_signal)
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _get_audio_sample_path(self,audio_dir):
        
        audio_sample_list=[]
        for file in os.listdir(audio_dir):
            path=os.path.join(audio_dir,file)
            audio_sample_list.append(path)
        return audio_sample_list
    
    def _get_audio_sample_label(self, audio_sample_name):
        data_dict=(self.annotations.loc[audio_sample_name])
        if isinstance(data_dict,pd.DataFrame):
            ans=data_dict.to_dict("records")
        else:
            
            ans=data_dict.to_dict()
            ans=[ans]
        ans=time_stamp_to_z(ans,313)
        return ans