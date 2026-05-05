
import pickle
import torch
import torch.nn as nn
from contrastive_model.complex_stuff import CplxLinear
import random
import numpy as np


# Load dataset
with open('guitarset_chord_embeddings_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f) # List of (recording_name, embedding, chord_label)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


CHORD_TO_IDX = {
    'N': 0,
    'C:maj': 1,
    'C:min': 2,
    'C#:maj': 3,
    'C#:min': 4,
    'D:maj': 5,
    'D:min': 6,
    'D#:maj': 7,
    'D#:min': 8,
    'E:maj': 9,
    'E:min': 10,
    'F:maj': 11,
    'F:min': 12,
    'F#:maj': 13,
    'F#:min': 14,
    'G:maj': 15,
    'G:min': 16,
    'G#:maj': 17,
    'G#:min': 18,
    'A:maj': 19,
    'A:min': 20,
    'A#:maj': 21,
    'A#:min': 22,
    'B:maj': 23,
    'B:min': 24
}


K = 5  # Number of folds
# Split dataset per recording in K folds
recordings = sorted(list(set([item[0] for item in dataset])))
random.shuffle(recordings)
fold_size = len(recordings) // K
blocks = []
for k in range(K):
    start_idx = k * fold_size
    if k == K - 1:
        end_idx = len(recordings)
    else:
        end_idx = (k + 1) * fold_size
    block_recordings = recordings[start_idx:end_idx]
    block_data = [item for item in dataset if item[0] in block_recordings]
    for idx in range(len(block_data)):
        datum = block_data[idx]
        label_idx = CHORD_TO_IDX.get(datum[2], 0)  # Default to 'N' if not found
        new_datum = (datum[0], torch.tensor(datum[1]), label_idx)  # Convert embedding to tensor
        block_data[idx] = new_datum

    blocks.append(block_data)

class ChordPredictionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=25, real_part=False):
        super(ChordPredictionHead, self).__init__()
        self.linear = CplxLinear(input_dim, output_dim)
        self.real_part = real_part

    def forward(self, x):
        x = self.linear(x)
        # Get magnitude, as chords are phase-invariant
        if self.real_part:
            x = x[:, 0]
        else:
            x = torch.sqrt((x**2).sum(1))
        return x # B x output_dim

# Cross-validation
all_accuracies = []
for k in range(K):
    print(f"Fold {k+1}/{K}")
    # Prepare train and test sets
    test_set = blocks[k]
    train_set = [item for i, block in enumerate(blocks) if i != k for item in block]

    # Create DataLoaders
    def collate_fn(batch):
        embeddings = torch.stack([item[1] for item in batch])  # B x D
        labels = torch.tensor([item[2] for item in batch])     # B
        return embeddings, labels

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = ChordPredictionHead(input_dim=512, output_dim=25, real_part=True).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(10):
        model.train()
        all_losses = []
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        avg_loss = np.mean(all_losses)
        print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}", end='\r')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.cuda(), labels.cuda()
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    all_accuracies.append(accuracy)
    print(f"Accuracy for fold {k+1}: {accuracy:.2f}%")

print(f"Average accuracy over {K} folds: {np.mean(all_accuracies):.2f}% ± {1.96*np.std(all_accuracies)/np.sqrt(K):.2f}%")