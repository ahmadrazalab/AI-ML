import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import Dataset

# Ensure GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 1. Load the dataset
def load_dataset():
    # Replace with your CSV file
    data = {
        "question": ["What is AI?", "Define machine learning?"],
        "context": [
            "Artificial intelligence (AI) refers to the simulation of human intelligence in machines.",
            "Machine learning is a subset of AI that focuses on training algorithms to learn from data."
        ],
        "answer": ["simulation of human intelligence", "training algorithms to learn from data"]
    }
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

# 2. Preprocess data
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples['context'],
        examples['question'],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(examples['answer']):
        context = examples['context'][i]
        answer_start = context.find(answer)
        answer_end = answer_start + len(answer)

        if answer_start == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            offsets = tokenized_inputs.offset_mapping[i]
            start_idx = end_idx = 0

            for idx, (start, end) in enumerate(offsets):
                if start <= answer_start < end:
                    start_idx = idx
                if start < answer_end <= end:
                    end_idx = idx
                    break

            start_positions.append(start_idx)
            end_positions.append(end_idx)

    tokenized_inputs['start_positions'] = start_positions
    tokenized_inputs['end_positions'] = end_positions
    return tokenized_inputs

# 3. Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# 4. Load and process datasets
data = load_dataset()
data = data.map(preprocess_data, batched=True, remove_columns=['context', 'question', 'answer'])
data = data.train_test_split(test_size=0.2)

train_dataset = data['train']
eval_dataset = data['test']

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # Updated as per FutureWarning
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=10,
    eval_steps=100,
    save_total_limit=2
)

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # Updated as per FutureWarning
    data_collator=data_collator
)

# 6. Train the model
trainer.train()

# 7. Save the model
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")

# 8. Inference function
def answer_question(context, question):
    inputs = tokenizer(
        context, question,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx], skip_special_tokens=True)
    return answer

# Example Usage
if __name__ == "__main__":
    context = "Artificial intelligence (AI) refers to the simulation of human intelligence in machines."
    question = "What is AI?"
    print("Answer:", answer_question(context, question))