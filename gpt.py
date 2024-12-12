from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from fastapi.responses import StreamingResponse
import time
from io import StringIO

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

# Function to clean up HTML from the answer
def clean_html(text: str):
    return BeautifulSoup(text, "html.parser").get_text()

# Streaming function to yield the response line by line
def generate_answer(question: str):
    relevant_context = find_relevant_context(question)
    clean_answer = clean_html(relevant_context)

    # Stream the answer line by line with a delay between each line to simulate real-time response
    for line in clean_answer.split("\n"):
        yield line + "\n"
        time.sleep(1)  # Optional delay to simulate typing

# API endpoint to answer a question
@app.post("/ask")
def ask_question(request: QARequest):
    # Create a StreamingResponse to send the answer line by line
    return StreamingResponse(generate_answer(request.question), media_type="text/plain")
