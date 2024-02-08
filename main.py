from fastapi import FastAPI, HTTPException
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
import torch
import re

app = FastAPI()

model_path = "model"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(model_path)

max_seq_length = 512

def preprocess_text(text):
    # Additional preprocessing steps can be added here if needed
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    return cleaned_text.strip()

def predict_dark_patterns(input_text):
    input_ids = tokenizer.encode(preprocess_text(input_text), return_tensors='pt', max_length=max_seq_length, truncation=True)

    with torch.no_grad():
        outputs = model(input_ids)

    probs = softmax(outputs.logits, dim=1).squeeze()
    predicted_category = torch.argmax(probs).item()

    return predicted_category

def count_dark_patterns(input_text):
    # Mapping category names to numeric labels
    category_mapping = {"Urgency": 0, "Not Dark Pattern": 1, "Scarcity": 2, "confirm shaming": 3, "Social Proof": 4,
                        "Obstruction": 5, "Sneaking": 6, "Forced Action": 7}

    dark_patterns = {category: 0 for category in category_mapping}

    sentences = re.split(r'[.!?]', input_text)

    for sentence in sentences:
        if not sentence.strip():
            continue

        category = predict_dark_patterns(sentence)
        category_name = next(key for key, value in category_mapping.items() if value == category)

        # Exclude "Not Dark Pattern" category
        if category_name != "Not Dark Pattern":
            dark_patterns[category_name] += 1

    return dark_patterns

@app.post("/detect-dark-patterns")
async def detect_dark_patterns(data: dict):
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Count dark patterns
    dark_patterns_count = count_dark_patterns(text)
    
    # Calculate percentage
    total_dark_patterns = sum(dark_patterns_count.values())
    percentage = (total_dark_patterns / len(text.split('.'))) * 100
    
    return {"dark_patterns_count": dark_patterns_count, "percentage": percentage}

@app.get("/")
async def root():
    return {"message": "Welcome to the Dark Patterns Detection API!"}