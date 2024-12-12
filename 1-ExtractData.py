import os
import yaml
import markdown
import pandas as pd
from bs4 import BeautifulSoup

# Function to extract YAML front matter and content from the MDX file
def extract_sections_from_mdx(mdx_file):
    with open(mdx_file, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # Extract YAML front matter (title, description)
    front_matter = None
    if md_content.startswith('---'):
        front_matter_end = md_content.find('---', 3)
        front_matter = md_content[3:front_matter_end].strip()
        md_content = md_content[front_matter_end + 3:].strip()

    # Parse front matter (YAML) into a dictionary
    front_matter_data = yaml.safe_load(front_matter) if front_matter else {}
    
    # Parse markdown content to HTML using markdown library
    html_content = markdown.markdown(md_content)

    return front_matter_data, html_content

# Function to extract Q&A pairs from the MDX file
def extract_qa_pairs(mdx_file):
    front_matter_data, html_content = extract_sections_from_mdx(mdx_file)

    qa_pairs = []
    title = front_matter_data.get('title', 'Unknown Title')
    description = front_matter_data.get('description', 'No description provided')

    # Add a basic question-answer pair based on title and description
    qa_pairs.append({
        "question": f"What is {title}?",
        "answer": description
    })

    # Parse the HTML content and extract headings (h1, h2, h3, etc.)
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3'])

    for heading in headings:
        heading_text = heading.get_text(strip=True)
        # Extract paragraph or text that follows the heading as the answer
        next_sibling = heading.find_next_sibling()
        answer_text = next_sibling.get_text(strip=True) if next_sibling else "Refer to the section for detailed explanation."

        # Handling code blocks in answers
        code_blocks = soup.find_all('code')
        for code_block in code_blocks:
            code_text = code_block.get_text(strip=True)
            answer_text += f"\n```hcl\n{code_text}\n```"

        qa_pairs.append({
            "question": f"What is {heading_text}?",
            "answer": answer_text
        })

    return qa_pairs

# Function to recursively scan directories for .mdx files and create a structured list of Q&A pairs
def create_qa_data_from_mdx_files(mdx_directory):
    all_qa_pairs = []

    # Recursively scan the directory for .mdx files
    for root, dirs, files in os.walk(mdx_directory):
        for file in files:
            if file.endswith(".mdx"):  # Only process .mdx files
                mdx_path = os.path.join(root, file)
                qa_pairs = extract_qa_pairs(mdx_path)
                all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs

# Store the data in a DataFrame and save as CSV (for chatbot use)
def save_qa_data_as_csv(mdx_directory, output_csv="qa_data.csv"):
    all_qa_pairs = create_qa_data_from_mdx_files(mdx_directory)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_qa_pairs)
    df.to_csv(output_csv, index=False)
    print(f"QA data saved to {output_csv}")

# Example: Set the path to your pages directory and call the function
mdx_directory = "./pages"
save_qa_data_as_csv(mdx_directory)
