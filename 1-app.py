from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace with any other model

# Load custom Q&A data
qa_data = pd.read_csv("qa_data.csv")  # Ensure this path points to your CSV file
qa_data = qa_data.fillna('')  # Fill missing data with empty strings

# Define the QARequest model for input validation
class QARequest(BaseModel):
    question: str

# Function to find the most relevant context for a question using Sentence Transformers
def find_relevant_context(question: str):
    best_match = None
    max_similarity = -1

    # Encode the question into an embedding
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Loop through each row in the dataset and find the most similar context
    for index, row in qa_data.iterrows():
        context = str(row['answer'])
        context_embedding = model.encode(context, convert_to_tensor=True)

        # Compute the cosine similarity between the question and the context
        similarity = util.pytorch_cos_sim(question_embedding, context_embedding)[0][0].item()

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = context

    if best_match is None:
        return "No relevant context found."

    return best_match

# API endpoint to answer a question
@app.post("/ask")
def ask_question(request: QARequest):
    # Find the most relevant context from the custom dataset
    relevant_context = find_relevant_context(request.question)

    if not relevant_context:
        return {"answer": "No relevant context found."}

    # If relevant context is found, perform inference using a pre-trained QA model
    # In this example, we use a simple approach since we're working with context-based questions

    # Use the SentenceTransformer model to re-encode the context for a better response
    context_embedding = model.encode(relevant_context, convert_to_tensor=True)
    question_embedding = model.encode(request.question, convert_to_tensor=True)

    # Compute the cosine similarity
    similarity_score = util.pytorch_cos_sim(question_embedding, context_embedding)[0][0].item()

    # If the similarity score is below a threshold, return no valid answer
    if similarity_score < 0.5:
        return {"answer": "No valid answer found based on the context."}

    # Return the best matching context as the answer
    return {"answer": relevant_context}

# To run the app, use the following command:
# uvicorn app:app --reload
