import os

from torchaudio.datasets import SPEECHCOMMANDS

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, root_dir='./root_dir', train=True, transform=None, download=True, labels = ['yes', 'no', 'up', 'down', 'left', 'right' ,'on' ,'off', 'stop', 'go']):
        super().__init__(root_dir, download=download)

        # Delete the downloaded .tar.gz file after extracting
        archive = os.path.join(root_dir, 'speech_commands_v0.02.tar.gz')
        if os.path.exists(archive):
            os.remove(archive)

        self.transform = transform
        self.labels = labels if labels is not None else []

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if train:
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        else:
            self._walker = load_list("testing_list.txt")

        self.label_to_int = {}
        for i, label in enumerate(self.labels):
            self.label_to_int[label] = i

        self.int_to_label = {}
        for i, label in enumerate(self.labels):
            self.int_to_label[i] = label

    def __getitem__(self, n: int):
        item = super().__getitem__(n)

        if self.transform is not None:
            waveform, sample_rate, *_ = item
            transformed_waveform = self.transform(waveform, sample_rate)
            item = [transformed_waveform, item[2]]

        return item

    def __iter__(self):
        for i in range(self.__len__()):
            item = self.__getitem__(i)
            if item[1] in self.labels:
                yield item

subsets = ['yes', 'no', 'up', 'down', 'left', 'right' ,'on' ,'off', 'stop', 'go']
sc = SubsetSC(root_dir='./root_dir', train=True, transform=None, download=True, labels=subsets)