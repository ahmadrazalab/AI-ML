import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

# Function to tokenize the extracted data
def preprocess_data(qa_data, tokenizer):
    inputs = []
    targets = []
    
    for question, answer in qa_data:
        # Tokenize the question and answer
        encoding = tokenizer(question, answer, truncation=True, padding="max_length", max_length=512)
        inputs.append(encoding['input_ids'])
        targets.append(encoding['attention_mask'])
    
    return inputs, targets

# Tokenizer for BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the extracted data
inputs, targets = preprocess_data(qa_data, tokenizer)

# Create a HuggingFace dataset from the preprocessed data
dataset = Dataset.from_dict({'input_ids': inputs, 'attention_mask': targets})

# Optionally, split dataset into train and eval sets
train_dataset = dataset.train_test_split(test_size=0.1)['train']
eval_dataset = dataset.train_test_split(test_size=0.1)['test']
