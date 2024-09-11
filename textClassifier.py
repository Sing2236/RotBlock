import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

# Load the scraped video metadata
with open('video_metadata.json', 'r') as f:
    video_metadata = json.load(f)

# Extract titles for text classification
titles = [item['title'] for item in video_metadata]
labels = [0 if "brainrot" in title.lower() else 1 for title in titles]  # Simple labeling logic

# Tokenize titles
tokenizer = get_tokenizer('basic_english')
tokenized_titles = [tokenizer(title) for title in titles]

# Build vocabulary from tokenized titles
def yield_tokens(data_iter):
    for text in data_iter:
        yield text

vocab = build_vocab_from_iterator(yield_tokens(tokenized_titles), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert tokenized titles to numerical form
def numericalize(text, vocab):
    return [vocab[word] for word in text]

numericalized_titles = [numericalize(title, vocab) for title in tokenized_titles]

# Define PyTorch dataset class
class YouTubeDataset(torch.utils.data.Dataset):
    def __init__(self, titles, labels):
        self.titles = titles
        self.labels = labels

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return torch.tensor(self.titles[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Create dataset and dataloader
dataset = YouTubeDataset(numericalized_titles, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Define text classification model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Model initialization
vocab_size = len(vocab)
embed_dim = 64
num_class = 2  # "brainrot" or "not brainrot"
model = TextClassificationModel(vocab_size, embed_dim, num_class)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (simplified)
for epoch in range(5):  # Example number of epochs
    total_loss = 0
    for titles, labels in dataloader:
        offsets = torch.tensor([0])  # Dummy offsets for EmbeddingBag
        output = model(titles, offsets)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')
