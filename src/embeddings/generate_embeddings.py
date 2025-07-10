import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import json

# -----------------------------
# 📍 Configuration
# -----------------------------
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
os.makedirs("data/embeddings", exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# -----------------------------
# 🔹 1. Embed utterances from top_40_intents_1000_samples.csv
# -----------------------------
print("📦 Embedding utterances...")

df_utterances = pd.read_csv("data/top_40_intents_1000_samples.csv")
text_column_1 = "phrase"
metadata_columns_1 = [
    "l3_intent", "l2_intent", "l1_intent",
    "category", "intent_type", "normalized_l3_intent"
]

utterance_texts = df_utterances[text_column_1].astype(str).tolist()
utterance_metadata = df_utterances[metadata_columns_1].to_dict(orient="records")
utterance_vectors = embedding_model.embed_documents(tqdm(utterance_texts))

np.save("data/embeddings/utterance_embeddings.npy", utterance_vectors)
with open("data/embeddings/utterance_texts.json", "w", encoding="utf-8") as f:
    json.dump(utterance_texts, f, indent=2)
with open("data/embeddings/utterance_metadata.json", "w", encoding="utf-8") as f:
    json.dump(utterance_metadata, f, indent=2)

print("✅ Utterance embeddings saved.")

# -----------------------------
# 🔹 2. Embed unique intent labels from normalized_l3_intent
# -----------------------------
print("\n📦 Embedding intent labels...")

unique_intents = sorted(df_utterances["normalized_l3_intent"].dropna().unique())
intent_vectors = embedding_model.embed_documents(tqdm(unique_intents))

np.save("data/embeddings/intent_label_embeddings.npy", intent_vectors)
with open("data/embeddings/intent_labels.json", "w", encoding="utf-8") as f:
    json.dump(unique_intents, f, indent=2)

print("✅ Intent label embeddings saved.")

# -----------------------------
# 🔹 3. Embed ground truth from ground_truth_dataset_top40_intent_30example.csv
# -----------------------------
print("\n📦 Embedding ground truth examples...")

df_ground = pd.read_csv("data/ground_truth_dataset_top40_intent_30example.csv")
text_column_2 = "Example phrase"
metadata_columns_2 = [
    "intent_definition", "l3_intent(master_intent)", "l2_intent", "l1_intent"
]

ground_texts = df_ground[text_column_2].astype(str).tolist()
ground_metadata = df_ground[metadata_columns_2].to_dict(orient="records")
ground_vectors = embedding_model.embed_documents(tqdm(ground_texts))

np.save("data/embeddings/ground_truth_embeddings.npy", ground_vectors)
with open("data/embeddings/ground_truth_texts.json", "w", encoding="utf-8") as f:
    json.dump(ground_texts, f, indent=2)
with open("data/embeddings/ground_truth_metadata.json", "w", encoding="utf-8") as f:
    json.dump(ground_metadata, f, indent=2)

print("✅ Ground truth embeddings saved.")
