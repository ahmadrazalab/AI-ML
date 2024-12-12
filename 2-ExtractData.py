import os
import json
import markdown

def extract_content(file_path):
    """ Extract content from mdx or md files, maintaining markdown syntax """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        # Use markdown to convert to HTML or just return the raw content
        html_content = markdown.markdown(file_content)
        return html_content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def generate_question_answer(dir_path, file_name):
    """ Generate question and answer based on directory and file content """
    question = f"What is {file_name.replace('-', ' ').title()}?"
    answer = extract_content(os.path.join(dir_path, file_name))
    return {"question": question, "answer": answer}

def walk_directory_and_generate_qa(root_dir):
    """ Walk the directory and generate question-answer pairs from all MDX/MD files """
    qa_pairs = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip non relevant files like _meta.json or images
        files = [f for f in files if f.endswith('.mdx') or f.endswith('.md')]
        
        # Generate Q&A pairs for each file
        for file in files:
            qa_pairs.append(generate_question_answer(root, file))
    
    return qa_pairs

def save_to_json(qa_pairs, output_file):
    """ Save the generated Q&A pairs to a JSON file """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

# Example usage:
root_directory = '/opt/pages'  # Path to your pages directory
output_file = '/opt/output/qa_pairs.json'  # Path to save the Q&A pairs

qa_pairs = walk_directory_and_generate_qa(root_directory)
save_to_json(qa_pairs, output_file)
