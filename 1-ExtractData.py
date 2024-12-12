import os
import markdown
from bs4 import BeautifulSoup

def parse_mdx_files(mdx_dir):
    data = []
    for root, _, files in os.walk(mdx_dir):
        for file in files:
            if file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    mdx_content = f.read()
                    html_content = markdown.markdown(mdx_content)
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for header in soup.find_all(['h1', 'h2', 'h3']):
                        question = header.get_text().strip()
                        answer = " ".join(
                            [p.get_text().strip() for p in soup.find_all('p') if p.find_previous('h1') == header]
                        )
                        data.append((question, answer))
    return data

mdx_dir = '/home/devops/Documents/ahmadrazalab/ahmadraza.in/docs.ahmadraza.in/pages'  # Update this path
qa_data = parse_mdx_files(mdx_dir)

# Save to a CSV file for inspection
import pandas as pd
df = pd.DataFrame(qa_data, columns=['question', 'answer'])
df.to_csv('qa_data.csv', index=False)
print("Data saved to qa_data.csv")
