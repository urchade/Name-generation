import numpy as np
from torch.utils.data import Dataset
from widis_lstm_tools.preprocessing import inds_to_one_hot


class NameDataset(Dataset):
    def __init__(self, filepath):

        self.pad_idx = 0
        self.name = []

        with open(filepath, 'r', encoding='utf8') as file:
            for name in file:
                self.name.append(name.strip('\n'))

        self.len_data = len(self.name)
        all_combined = ''.join(self.name)

        # 0 for padding, 1 for <sos>, 2 for <eos>
        self.id2char = dict(enumerate(set(all_combined), start=3))
        self.id2char[0] = '<pad>'
        self.id2char[1] = '<sos>'
        self.id2char[2] = '<eos>'

        self.char2id = {v: k for k, v in self.id2char.items()}

        self.id_names = [[self.char2id[char] for char in name] for name in
                         self.name]

        self.seq_len = max([len(n) for n in self.id_names]) + 2

        for i in range(self.len_data):
            self.id_names[i].append(2)  # Insert <eos> at the end
            self.id_names[i].insert(0, 1)  # Insert <sos> at the start
            self.id_names[i] = self.pad_sequence(self.id_names[i]) # padding to max lenght

    def __len__(self):
        return len(self.id_names)

    def __getitem__(self, item):

        x = np.array(self.id_names[item])
        one_hot_x = inds_to_one_hot(x, len(self.id2char))

        y = np.zeros_like(self.id_names[item])
        y[:-1] = x[1:]

        return one_hot_x, y

    def pad_sequence(self, list_ids):
        """
        Parameters
        ----------
        list_ids: A list of sequence of ids

        Returns
        -------
        Padded array
        """
        padded_array = np.ones(self.seq_len, dtype=np.int) * self.pad_idx
        for i, el in enumerate(list_ids[:self.seq_len]):
            padded_array[i] = el
        return padded_array
