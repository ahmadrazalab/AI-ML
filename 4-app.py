from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from bs4 import BeautifulSoup  # Import BeautifulSoup for HTML parsing

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",  # Allow frontend running on localhost
    "http://localhost:3000",  # If your frontend runs on port 3000
    "http://yourfrontenddomain.com",  # Add your production frontend domain if applicable
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
    action: str = None  # 'action' can be 'files' to show files within folders

# Function to find relevant folders or files based on the user query
def find_relevant_folders(query: str):
    matching_folders = []
    for index, row in qa_data.iterrows():
        folder_name = row['answer']  # Folder name is the 'answer' field
        if query.lower() in folder_name.lower():  # Case-insensitive partial match
            matching_folders.append(folder_name)
    return matching_folders

def list_files_in_folder(folder_path: str):
    """ List all files in the folder """
    files = []
    for root, dirs, files_in_dir in os.walk(folder_path):
        for file in files_in_dir:
            if file.endswith('.mdx') or file.endswith('.md'):
                files.append(os.path.join(root, file))
    return files

@app.post("/ask")
def ask_question(request: QARequest):
    query = request.question
    action = request.action

    # Find matching folders
    matching_folders = find_relevant_folders(query)

    if not matching_folders:
        return {"answer": "No relevant folders found."}

    if action == "files":
        # If user wants to see files, list files in the matching folders
        files_in_folders = {}
        for folder in matching_folders:
            files_in_folders[folder] = list_files_in_folder(folder)

        return {"answer": files_in_folders}

    # Return the matching folders as the answer
    return {"answer": matching_folders}

