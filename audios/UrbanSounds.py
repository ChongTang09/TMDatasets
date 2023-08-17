import os
import torch
import torchaudio
import subprocess
from torch.utils.data import Dataset

class UrbanSound8K(Dataset):
    def __init__(self, root = './root_dir', transform = None, train = True, test_fold = 1, download=True):
        
        self.root_dir = root
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.transform = transform

        # Check if the data is present, if not, download and extract it.
        if download and not os.path.exists(self.root_dir + '/UrbanSound8K'):
            print("Downloading the UrbanSound8K dataset...")
            subprocess.call(['wget', 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz', '-O', 'urban8k.tgz'])
            print("Extracting the UrbanSound8K dataset...")
            subprocess.call(['tar', '-xzf', 'urban8k.tgz'])
            subprocess.call(['rm', 'urban8k.tgz'])

        self.data = os.listdir(self.root_dir)
        self.test_fold = str(test_fold)

        self.walker = []
        for filename in self.data:
            if train:
                if 'fold' in filename: 
                    if filename != ('fold' + self.test_fold):
                        for j in os.listdir(os.path.join(self.root_dir,filename)):
                            if j!='.DS_Store':
                                self.walker.append(os.path.join(self.root_dir,filename,j))
            else: 
                if 'fold' in filename: 
                    if filename == ('fold' + self.test_fold):
                        for j in os.listdir(os.path.join(self.root_dir,filename)):
                            if j!='.DS_Store':
                                self.walker.append(os.path.join(self.root_dir,filename,j))
    def __len__(self):
        return len(self.walker)
    
    def __getitem__(self, index):
        fname = self.walker[index]
        splits_fname = fname.split('/')[-1].split('-')
        label = splits_fname[1]    

        waveform, sampling_rate = torchaudio.load(fname)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  
        print(waveform.shape, sampling_rate)    
        
        if self.transform:
            features = self.transform(waveform, sampling_rate)
            return features, label
        else:
            return waveform, label

us = UrbanSound8K(root = './root_dir', transform = None, train = True, test_fold = 1, download=True)