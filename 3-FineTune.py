import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

# Load extracted data
data = pd.read_csv('qa_data.csv')
dataset = Dataset.from_pandas(data)

# Split dataset into training and evaluation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    inputs = examples['question']
    targets = examples['answer']
    
    # Debugging logs
    print(f"Type of inputs: {type(inputs)}, Content: {inputs}")
    print(f"Type of targets: {type(targets)}, Content: {targets}")
    
    # Ensure both inputs and targets are strings or lists of strings
    if isinstance(inputs, list):
        inputs = [str(inp) for inp in inputs]
    else:
        inputs = str(inputs)
        
    if isinstance(targets, list):
        targets = [str(tgt) for tgt in targets]
    else:
        targets = str(targets)
    
    model_inputs = tokenizer(inputs, targets, truncation=True, padding="max_length", max_length=512)
    return model_inputs


# Preprocess datasets
train_dataset = train_dataset.map(preprocess_data, batched=True)
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# Initialize model
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
print("Model fine-tuned and saved to ./fine_tuned_model")
