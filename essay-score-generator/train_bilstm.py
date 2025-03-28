import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import numpy as np

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define Bi-LSTM Model
class BiLSTMGrammarModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMGrammarModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])  # Taking output from last time step
        return logits

# Model Hyperparameters
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
output_dim = 1

# Initialize Model
model = BiLSTMGrammarModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Training Data (Dummy Example)
train_sentences = ["This is a good essay.", "The grammar of this sentence is bad.", "Perfectly structured essay."]
train_labels = torch.tensor([1.0, 0.0, 1.0]).unsqueeze(1)  # 1 = Good Grammar, 0 = Bad Grammar

# Tokenize Sentences
tokenized_sentences = [tokenizer.encode(s, truncation=True, max_length=512, padding="max_length") for s in train_sentences]
train_inputs = torch.tensor(tokenized_sentences)

# Loss Function & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
for epoch in range(100):  # 100 epochs
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/100] - Loss: {loss.item():.4f}")

# Save Model
torch.save(model.state_dict(), "bilstm_model.pth")
print("✅ Model trained and saved as `bilstm_model.pth`")
