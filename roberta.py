import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset

# Step 1: Load and Preprocess SST-2 Data
dataset = load_dataset("glue", "sst2")
train_data = dataset['train']
test_data = dataset['validation']

# Load tokenizer and pre-process the data for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Convert labels
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Step 2: Define a Probing Sequence Function
def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

def evaluate_prompt_variations(prompt_variations, model, tokenizer, test_data):
    entropies = []
    for prompt in prompt_variations:
        pipeline_model = pipeline("text-classification", model=model, tokenizer=tokenizer)
        predictions = pipeline_model(prompt)
        
        probs = np.array([pred["score"] for pred in predictions])
        entropy = compute_entropy(probs)
        entropies.append((prompt, entropy))
    # Select the top-k prompts by lowest entropy
    top_k_prompts = sorted(entropies, key=lambda x: x[1])[:4]
    return top_k_prompts

# Step 3: Load and Train RoBERTa
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Step 4: Evaluate RoBERTa Performance on Test Data
roberta_results = trainer.evaluate()

# Step 5: Compare Probing Sequence Performance with RoBERTa
prompt_variations = ["Review: {sentence}", "Input: {sentence} Sentiment:", "Prediction: {sentence}"]
selected_prompts = evaluate_prompt_variations(prompt_variations, model, tokenizer, test_data)

print("RoBERTa Results:", roberta_results)
print("Top Probing Prompts:", selected_prompts)
