import os
import torch
import random
import zipfile
import numpy as np
import pandas as pd
import urllib.request
from collections import Counter

from torch.utils.data import Dataset

from Binarizer import Binarizer

def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]

def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]

def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output

def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output
    
def load_subjects(root='data/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    
    for i in range(101,110):
        path = root + str(i) +'.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols 
        subject['id'] = i
        output = pd.concat([output, subject], ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output

class PAMAP2Dataset(Dataset):

    def __init__(self, root_dir='./root_dir', train=True, binarize=True, lstm=False, 
                more_features=False, download=True, train_size_per_class=40000, test_size_per_class=10000):
        self.root_dir = root_dir
        self.train = train
        self.lstm = lstm
        self.train_size = train_size_per_class
        self.test_size = test_size_per_class
        
        if download:
            self.download_dataset()

        self.activity_map = {
            0: 'transient', 1: 'lying', 2: 'sitting', 3: 'standing',
            4: 'walking', 5: 'running', 6: 'cycling', 7: 'Nordic_walking',
            9: 'watching_TV', 10: 'computer_work', 11: 'car driving', 12: 'ascending_stairs',
            13: 'descending_stairs', 16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
            19: 'house_cleaning', 20: 'playing_soccer', 24: 'rope_jumping'
        }

        # Process data
        self.data = load_subjects(root=self.root_dir + '/PAMAP2_Dataset/Protocol/subject')
        self.data = self.fix_data(self.data)
        if more_features:
            self.data = self.add_more_features(self.data)

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(self.data)

        # Define conversion dictionaries based on unique labels in the dataset
        unique_labels = sorted(np.unique(np.concatenate((self.y_train, self.y_test))))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Convert y_train and y_test labels to indices
        self.y_train = np.array([self.label_to_index[label] for label in self.y_train])
        self.y_test = np.array([self.label_to_index[label] for label in self.y_test])
        
        if binarize:
            binarizer = Binarizer(max_bits_per_feature=10)
            binarizer.fit(self.X_train)
            self.X_train = binarizer.transform(self.X_train)
            self.X_test = binarizer.transform(self.X_test)
        
        if lstm:
            self.X_train, self.y_train = self.create_lstm_data(self.X_train, self.y_train)
            self.X_test, self.y_test = self.create_lstm_data(self.X_test, self.y_test)

    def fix_data(self, data):
        data = data.drop(data[data['activity_id']==0].index)
        data = data.interpolate()
        for colName in data.columns:
            data[colName] = data[colName].fillna(data[colName].mean())
        return data
    
    def create_lstm_data(self, X, y, step_back=5, step_forward=1):
        out_X = []
        out_y = []
        size = len(X)
        for i, features in enumerate(X):
            if i >= step_back and i < size - step_forward:
                tmp_X = []
                tmp_y = []
                for j in range(i-step_back,i):
                    tmp_X.extend([X[j]])
                out_X.append(tmp_X)
                for j in range(i,i+step_forward):
                    tmp_y.extend([y[j]])
                out_y.append(tmp_y)
        return np.array(out_X), np.array(out_y)

    def add_more_features(self, data):
        new_data = data.copy().reset_index()
        new_cols = None 
        for subject in range(101,110):
            prev_act_1 = new_data[new_data['id'] == subject]
            start = prev_act_1.head(2).index[1]
            end = prev_act_1.tail(1).index[0]
            prev_act_1 = prev_act_1.loc[start:end+1]
            new_cols_1 = pd.DataFrame()
            new_cols_1['prev_aid'] = prev_act_1['activity_id']
            new_cols_1['prev_hr'] = prev_act_1['heart_rate']
            new_cols_1['index'] = prev_act_1['index'] + 1
            if new_cols is None:
                new_cols = new_cols_1
            else:
                new_cols = pd.concat([new_cols, new_cols_1], ignore_index=True)
        new_cols = new_data.merge(new_cols, on='index', how='left')
        new_cols = new_cols.dropna()
        return new_cols
    
    def split_train_test(self, data):
        # Split data
        subject107 = data[data['id'] == 107]
        subject108 = data[data['id'] == 108]
        test = pd.concat([subject107, subject108])
        train = data[data['id'] != 107]
        train = data[data['id'] != 108]

        # Drop columns
        test = test.drop(["id"], axis=1)
        train = train.drop(["id"], axis=1)
        X_train = train.drop(['activity_id','time_stamp'], axis=1).values
        X_test = test.drop(['activity_id','time_stamp'], axis=1).values
        
        y_train = train["activity_id"].values
        y_test = test["activity_id"].values
        
        # If train_size is set, sample from X_train and y_train
        if self.train_size:
            total_train_size = self.train_size * len(np.unique(y_train))
            X_train, y_train = self.balanced_sample(X_train, y_train, total_train_size)
            
        # If test_size is set, sample from X_test and y_test
        if self.test_size:
            total_test_size = self.test_size * len(np.unique(y_test))
            X_test, y_test = self.balanced_sample(X_test, y_test, total_test_size)

        return X_train, X_test, y_train, y_test

    def balanced_sample(self, X, y, sample_size):
        class_counts = Counter(y)
        classes = list(class_counts.keys())
        samples_per_class = sample_size // len(classes)

        new_X = []
        new_y = []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            random.shuffle(cls_indices)

            # Ensure we don't try to sample more items than exist
            samples_from_this_class = min(len(cls_indices), samples_per_class)
            new_X.extend(X[cls_indices[:samples_from_this_class]])
            new_y.extend(y[cls_indices[:samples_from_this_class]])

        return np.array(new_X), np.array(new_y)
    
    def download_dataset(self):
        # Define the download URL and paths for expected directories and files
        url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
        save_path = os.path.join(self.root_dir, "pamap2.zip")
        dataset_dir = os.path.join(self.root_dir, "PAMAP2_Dataset")
        protocol_dir = os.path.join(dataset_dir, "Protocol")

        # Check if dataset already exists
        if os.path.exists(protocol_dir):
            print("Dataset already exists. Skipping download.")
            return

        # If not, check if the main zip already exists, if not download it
        if not os.path.exists(save_path):
            print("Downloading the dataset...")
            urllib.request.urlretrieve(url, save_path)

        # Extract the main zip
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            print("Extracting the main zip...")
            zip_ref.extractall(self.root_dir)

        # Remove the readme.pdf after extracting main zip
        readme_path = os.path.join(self.root_dir, "readme.pdf")
        if os.path.exists(readme_path):
            os.remove(readme_path)

        # Confirm extraction of nested zip
        nested_zip_path = os.path.join(self.root_dir, "PAMAP2_Dataset.zip")
        if os.path.exists(nested_zip_path):
            print("Found the nested zip: PAMAP2_Dataset.zip")
            with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                print("Extracting the nested zip...")
                zip_ref.extractall(self.root_dir)
            os.remove(nested_zip_path)  # remove the nested zip after extraction
        else:
            print("Couldn't find the nested zip: PAMAP2_Dataset.zip")

        # Remove the main zip
        os.remove(save_path)

    def __len__(self):
        return len(self.X_train) if self.train else len(self.X_test)
    
    def __getitem__(self, idx):
        if self.train:
            sample = (self.X_train[idx], self.y_train[idx])
        else:
            sample = (self.X_test[idx], self.y_test[idx])
        
        return sample