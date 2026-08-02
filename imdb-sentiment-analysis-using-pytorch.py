import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gensim.downloader import load

df = pd.read_csv("IMDB Dataset.csv")


df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


df = df.sample(frac=1, random_state=42)



print("Loading GloVe model...")

word_vectors = load("glove-wiki-gigaword-300")
EMBED_DIM = 300
print("Model loaded successfully!")


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


def sentence_vector(sentence):

    sentence = clean_text(sentence)
    words = sentence.split()

    vectors = []

    negate = False
    after_but = False

    for word in words:

        # BUT handling
        if word == "but":
            after_but = True
            continue

        # negation handling
        if word in ["not", "no", "never", "n't"]:
            negate = True
            continue

        if word in word_vectors:

            vec = word_vectors[word]

            # improved negation (soft inversion instead of full flip)
            if negate:
                vec = vec * -0.3
                negate = False

            # BUT context (slightly more importance, not ×2)
            if after_but:
                vec = vec * 1.3

            vectors.append(vec)

    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)

    return np.mean(vectors, axis=0)


X = np.array([sentence_vector(text) for text in df["review"]])

y = np.array(df["sentiment"])



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)


class SentimentModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)



model = SentimentModel()



criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)



epochs = 150

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")



with torch.no_grad():

    predictions = model(X_test)

    predictions = (predictions >= 0.5).float()
    
    
    accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nFinal Accuracy:", accuracy)



model.eval()

while True:
    user_input = input("\nEnter a review: ")

    vec = sentence_vector(user_input)

    vec = torch.tensor(vec, dtype=torch.float32)
    
    vec = vec.unsqueeze(0)

    with torch.no_grad():
        
        prediction = model(vec)
        
        value = prediction.item()
        
    print(f"Sentiment Score: {value:.4f}")

    if value >= 0.5:
        
        print("Positive Review")
        
    else:
        
        print("Negative Review")