from transformers import pipeline

# Load the fine-tuned model
qa_pipeline = pipeline('question-answering', model='./fine_tuned_model', tokenizer='bert-base-uncased')

# Example context and question
context = "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines."
question = "What is AI?"

result = qa_pipeline(question=question, context=context)
print(result)
