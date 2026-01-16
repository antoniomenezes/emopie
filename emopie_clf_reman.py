import os

# EMOPIE classifier trained with REMAN preprocessed dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
)

# Datasets directory
data_dir = "./data"

# Constant
SEED_ALL = 42

# Setting up the device for GPU usage if available
device = "cuda" if cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()
print(n_gpu)
print(torch.cuda.get_device_name(0))

import spacy

nlp = spacy.load("en_core_web_sm")


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED_ALL)

# Training Emotion Classifier

# Loading preprocessed dataset
df_reman = pd.read_csv(
    data_dir + "/" + "REMAN_multilabel.csv", sep="|", encoding="utf-8"
)

df_reman['emotion_count'] = df_reman['emotion_count'].astype(int)
df_reman['entity_start'] = df_reman['entity_start'].astype(int)
df_reman['entity_end'] = df_reman['entity_end'].astype(int)
df_reman['emotion_labels'] = df_reman['emotion_labels'].apply(eval)
df_reman['emotions'] = df_reman['emotions'].apply(eval)

df_train, df_test = train_test_split(df_reman, random_state=SEED_ALL, stratify=df_reman['emotion_count'], test_size=0.2)

# Variables for training
emotion_label_names = sorted(set([emotion for sublist in df_train['emotions'] for emotion in sublist]))
N_CLASSES = len(emotion_label_names)
print(N_CLASSES)

# Multi-label classification dataset
class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, num_classes):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        start = item['entity_start']
        end = item['entity_end']
        emotion_ids = item['emotion_labels']  # ID list

        tokens = self.tokenizer(text, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=self.max_len)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        entity_mask = torch.zeros_like(input_ids)
        if 0 <= start < self.max_len and 0 <= end <= self.max_len and start < end:
            entity_mask[start:end] = 1

        multi_hot_label = torch.zeros(self.num_classes, dtype=torch.float)
        for eid in emotion_ids:
            if 0 <= eid < self.num_classes:
                multi_hot_label[eid] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_mask': entity_mask,
            'emotion_label': multi_hot_label
        }

# Classification Head
class EmotionClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, entity_emb, context_emb):
        x = torch.cat([entity_emb, context_emb], dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# The Entity Emotion Model
class EntityEmotionModel(nn.Module):
    def __init__(self, bert, classifier):
        super().__init__()
        self.bert = bert
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, entity_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        context_emb = out.pooler_output
        token_embs = out.last_hidden_state

        entity_emb = (token_embs * entity_mask.unsqueeze(-1)).sum(1)
        entity_emb = entity_emb / entity_mask.sum(1).unsqueeze(1).clamp(min=1e-6)

        logits = self.classifier(entity_emb, context_emb)
        return logits

# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        labels = batch['emotion_label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, entity_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, device, label_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask = batch['entity_mask'].to(device)
            labels = batch['emotion_label'].to(device)

            logits = model(input_ids, attention_mask, entity_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.4).int() # before, 0.5

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }, y_true, y_pred


# Multi-Label Confusion Matrix
def plot_multilabel_confusion_matrix(
    y_true, y_pred, label_names, figsize=(10, 5), cmap="Grays"
):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    n_labels = len(label_names)
    cols = 3
    rows = (n_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_labels):
        cm = mcm[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=axes[i],
            xticklabels=["Not " + label_names[i], label_names[i]],
            yticklabels=["Not " + label_names[i], label_names[i]],
        )
        axes[i].set_title(f"'{label_names[i]}' Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    # Turn off empty subplots
    for i in range(n_labels, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Configuration
EPOCHS = 4
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)
classifier = EmotionClassificationHead(bert_model.config.hidden_size, N_CLASSES)
model = EntityEmotionModel(bert_model, classifier).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()
train_dataset = EmotionDataset(
    df_train.to_dict("records"), tokenizer, max_len=512, num_classes=N_CLASSES
)
test_dataset = EmotionDataset(
    df_test.to_dict("records"), tokenizer, max_len=512, num_classes=N_CLASSES
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training loop
print("\nTraining...")

for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

results, y_true, y_pred = evaluate(model, test_loader, device, emotion_label_names)
print(results)

# Plotting Multilabel Confusion Matrix
plot_multilabel_confusion_matrix(y_true, y_pred, emotion_label_names)
