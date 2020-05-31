import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from widis_lstm_tools.preprocessing import random_dataset_split, inds_to_one_hot

from dataset import NameDataset
from model import NameLSTM
from train import train

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load dataset
filepath = r'data\Arabic.txt'
data = NameDataset(filepath)

# data splitting
train_data, test_data = random_dataset_split(data, split_sizes=(90 / 100., 10 / 100))

# data loader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# model
model = NameLSTM(n_inputs=len(data.id2char), hidden_size=128)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Optimizer, scheduler
optimizer = torch.optim.Adam(model.parameters(), 0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[20, 30, 40, 60],
                                                 gamma=0.3)
# training
train(model, optimizer, criterion, 50, train_loader,
      test_loader, scheduler=scheduler)

# Setting to cpu and evaluation model (for deactivating dropout)
model.cpu()
model.eval()


# Function that transforms and index to a one-hot tensor
@torch.no_grad()
def transform(idx):
    # one_hot encoding
    idx = inds_to_one_hot(np.array(idx), len(data.id2char))
    # (batch_size, seq_len, input_size)
    idx = torch.Tensor(idx).reshape(1, 1, len(data.id2char))
    return idx


@torch.no_grad()
def generate_one_name(top_k=2):
    """
    Function for generating one molecule
    """
    # 0 is the <sos> id
    x = transform(1)

    out, hidden_states = model.forward(x)

    # Get the top k logits its corresponding ids
    topk = torch.topk(out, k=top_k)

    # Get the top k probabilities
    topk_prob = torch.softmax(topk[0].squeeze(), dim=-1).numpy()

    # Get the top k ids
    topk_ids = topk[1].squeeze().numpy()

    # Sampling
    pred = np.random.choice(topk_ids, p=topk_prob)

    # the predicted character is the next input
    inputs = transform(pred)

    # the first element of our molecule is the first prediction
    molecule = [data.id2char[pred]]

    # Generating one character while <eos> is not reached
    while True:
        # predict next char
        out, hidden_states = model.forward(inputs, hidden_states)

        # same process as before ...
        topk = torch.topk(out, k=top_k)

        topk_prob = torch.softmax(topk[0].squeeze(), dim=-1).numpy()

        topk_ids = topk[1].squeeze().numpy()

        out_id = np.random.choice(topk_ids, p=topk_prob)

        # Break if we reach the <eos> token
        if out_id == 2 or out_id == 0:
            break

        # Decode the prediction and append it to the list of characters
        molecule.append(data.id2char[out_id])

        # transform the next input
        inputs = transform(out_id)

    # Join all the generated character to make the molecule
    return ''.join(molecule)


def generate_many_names(n=10000, top_k=5):
    """
    Function that return a list of n generated molecules
    """
    list_generated = []
    for _ in tqdm(range(n)):
        while True:
            # Generate one molecule
            name = generate_one_name(top_k=top_k)

            # Test if the generated molecule is valid and is not in the training set
            if name not in list_generated and name not in data.name:
                # If valid, we add the generated molecule to our list and
                # break to go to the next iteration.
                list_generated.append(name)
                break
    return list_generated


# generating 10000 molecules
generated = generate_many_names(30, top_k=5)

