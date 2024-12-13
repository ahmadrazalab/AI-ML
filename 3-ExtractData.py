import os
import pandas as pd

def generate_question_answer(dir_path, file_name):
    """ Generate question and answer based on directory and file content """
    # Extract the topic from the file name (or directory name)
    topic = file_name.replace('-', ' ').split('.')[0]  # We use file name without extension as the topic
    
    # Replace the local directory path with the desired URL format
    base_url = "docs.ahmadraza.in"  # Base URL for the documentation site
    relative_path = os.path.relpath(dir_path, start='/opt/pages')  # Get the relative path from the root directory
    answer = f"{base_url}/{relative_path.replace(os.sep, '/')}"

    return {"question": topic, "answer": answer}

def walk_directory_and_generate_qa(root_dir):
    """ Walk the directory and generate question-answer pairs from all MDX/MD files """
    qa_pairs = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip non-relevant files like _meta.json or images
        files = [f for f in files if f.endswith('.mdx') or f.endswith('.md')]
        
        # Generate Q&A pairs for each file
        for file in files:
            qa_pairs.append(generate_question_answer(root, file))
    
    return qa_pairs

def save_to_csv(qa_pairs, output_file):
    """ Save the generated Q&A pairs to a CSV file """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(qa_pairs)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

# Example usage:
root_directory = '/opt/pages'  # Path to your pages directory
output_file = '/opt/qa_data.csv'  # Path to save the Q&A pairs

qa_pairs = walk_directory_and_generate_qa(root_directory)
save_to_csv(qa_pairs, output_file)
