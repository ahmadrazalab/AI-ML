from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",  # Allow frontend running on localhost
    "http://localhost:3000",  # If your frontend runs on port 3000
    "https://docs.ahmadraza.in",  # Add your production frontend domain if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows CORS for these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, you can restrict to specific methods like ["GET", "POST"]
    allow_headers=["*"],  # Allow all headers
)

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
    best_matches = []
    max_similarity = -1

    # Encode the question into an embedding
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Loop through each row in the dataset and find the most similar contexts (directories)
    for index, row in qa_data.iterrows():
        context = str(row['answer'])  # This is the directory path
        context_embedding = model.encode(context, convert_to_tensor=True)

        # Compute the cosine similarity between the question and the context
        similarity = util.pytorch_cos_sim(question_embedding, context_embedding)[0][0].item()

        if similarity > max_similarity:
            max_similarity = similarity
            best_matches = [context]  # Found a new best match
        elif similarity == max_similarity:
            best_matches.append(context)  # Add to list if the similarity is the same

    if not best_matches:
        return "No relevant context found."

    return best_matches

# API endpoint to answer a question
@app.post("/ask")
def ask_question(request: QARequest):
    # Find the most relevant context (directories) for the question
    relevant_context = find_relevant_context(request.question)

    if not relevant_context:
        return {"answer": "No relevant context found."}

    # Return the matching directories
    return {"answer": ", ".join(relevant_context)}
