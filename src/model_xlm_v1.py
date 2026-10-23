import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv(r'C:\Users\ramaz\Desktop\diplom\data\id\language_id.csv')
# print(df.head())
# print(df["language"].value_counts())
df = df.dropna(subset=["text","language"]).copy()

df["text"] = df["text"].astype(str)
df["language"] = df["language"].astype(str)

df["text"] = df["text"].str.strip()
df["language"] = df["language"].str.strip()

df = df[
    (df["text"] != "") & 
    (df["language"] != "") 

]

df = df.drop_duplicates(subset = ["text","language"])

# print("dataset size after cleanin",len(df))
labels = sorted(df["language"].unique())

labeltoid = {label:i for i,label in enumerate(labels)}
idtolabel = {i:label for label,i in labeltoid.items()}
df["label_id"] = df["language"].map(labeltoid)
# print("Label mapping:", labeltoid)
# print(df[["text", "language", "label_id"]].head())

num_labels = len(labeltoid)
# print(num_labels)
model_name = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ldataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = "max_length",
            max_length=self.max_length,
            return_tensors = "pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }



dataset = ldataset(
    texts=df["text"].tolist(),
    labels=df["label_id"].tolist(),
    tokenizer=tokenizer,
    max_length=128
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("Training examples:", len(dataset))
print("Batches per epoch:", len(train_loader))



model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=idtolabel,
    label2id=labeltoid
)

loss_fn = torch.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr = 2e-5) 

model.train()

for i in range(10):

    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        att_mask = batch["attention_mask"]
        labels = batch["labels"]

        optimizer.zero_grad()

        outputs = model(
            input_ids = input_ids,
            attention_mask = att_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits,labels)
    

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {i + 1}/10, Loss: {avg_loss:.4f}")

test_sentences = [
    "Бүгін ауа райы өте жақсы.",
    "Мен университетке барып келдім.",
    "Қазақстанның астанасы — Астана.",
    "Маған машиналық оқыту қызық.",
    "Ертең біз кітапханаға барамыз.",
    "Сегодня погода очень хорошая.",
    "Я сегодня ходил в университет.",
    "Столица России — Москва.",
    "Мне интересно машинное обучение.",
    "Завтра мы пойдем в библиотеку."
]

model.eval()

for sentence in test_sentences:
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        predicted_language = idtolabel[prediction]

        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item()

    print(f"Text: {sentence}")
    print(f"Predicted language: {predicted_language}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 50)