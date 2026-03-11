import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm


class FinBERTSentiment:

    def __init__(self, batch_size=16):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        self.batch_size = batch_size

    def score_batch(self, texts):

        scores = []

        for i in tqdm(range(0, len(texts), self.batch_size)):

            batch_texts = texts[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = F.softmax(outputs.logits, dim=1).numpy()

            batch_scores = probs[:, 2] - probs[:, 0]
            scores.extend(batch_scores)

        return scores