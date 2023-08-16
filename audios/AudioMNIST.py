import os
import subprocess
import numpy as np

from torch.utils.data import Dataset

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir='./root_dir', download=True, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            train (bool): If True, returns the training dataset, else returns the test dataset.
            transform (callable, optional): Optional transform to be applied on an audio sample.
            test_split (float): Proportion of the dataset to be used as test data. Default is 0.2.
            seed (int): Random seed to ensure reproducibility.
        """
        self.root = root_dir
        self.transform = transform

        self.data = []
        self.targets = []

        self.splits = {"train":[   set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                                          8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, \
                                         10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, \
                                          4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, \
                                          5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1, \
                                          6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54])],

                    "validate":[set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])],

                    "test":[    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])]}

        if download:
            self.download()

        # Load the dataset
        self.load_dataset(train)

    def load_dataset(self, train=True, rnn=False):        
        # Get the training and testing ids based on the first split set
        train_ids = self.splits['train'][0].union(self.splits['validate'][0])  # Combine train and validate
        test_ids = self.splits['test'][0]
        
        # Loop through each subfolder (person id)
        for person_id in sorted(os.listdir(self.root)):
            if person_id.isnumeric():  # This will ensure we only handle folders named with numbers.
                person_id_num = int(person_id)  # Convert '01', '02', etc. to integers
                person_folder = os.path.join(self.root, person_id)

                if os.path.isdir(person_folder):
                    for filename in sorted(os.listdir(person_folder)):
                        if filename.endswith('.wav'):
                            label, _, _ = filename.split('_')
                            waveform, sample_rate = torchaudio.load(os.path.join(person_folder, filename))
                            
                            # Apply the transform to standardize the waveform length and compute the MFCC features
                            waveform = self.transform(waveform, sample_rate) # Suggest MFCC transform
                            # Assign data and label based on person_id and whether we're in training or testing mode
                            if train and person_id_num in train_ids:
                                self.data.append(waveform)
                                self.targets.append(int(label))
                            elif not train and person_id_num in test_ids:
                                self.data.append(waveform)
                                self.targets.append(int(label))
        
        # Convert lists to numpy arrays
        self.data = np.array([tensor.numpy() for tensor in self.data])
        self.targets = np.array(self.targets)

    def download(self):
        target_path = os.path.join(self.root, "AudioMNIST")
        if not os.path.exists(target_path):
            print('Downloading AudioMNIST dataset...')
            subprocess.check_call(['git', 'clone', 'https://github.com/soerenab/AudioMNIST.git', target_path])
        else:
            print("Dataset already downloaded!")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        label = self.targets[idx]
        return [waveform, label]

audiomnist = AudioMNISTDataset(root_dir='./root_dir', download=True, train=True, transform=None)