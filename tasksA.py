import sqlite3
import subprocess
from dateutil.parser import parse
from datetime import datetime
import json
from pathlib import Path
import os
import requests
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from fastapi import HTTPException
import shutil

load_dotenv()

AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')


def A1(email="22f2001184@ds.study.iitm.ac.in"):
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")

def A2(prettier_version="prettier@3.4.2", filename="./data/format.md"):
    npx_path = shutil.which("npx") 
    if not npx_path:
        print("Error: npx not found. Ensure Node.js is installed.")
        return
    
    command = [npx_path, prettier_version, "--write", filename]
    
    try:
        subprocess.run(command, check=True)
        print("Prettier executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")



def A3(filename='/data/dates.txt', targetfile='/data/dates-wednesdays.txt', weekday=2):
    input_file = filename
    output_file = targetfile
    weekday = weekday
    weekday_count = 0

    with open(input_file, 'r') as file:
        weekday_count = sum(1 for date in file if parse(date).weekday() == int(weekday)-1)


    with open(output_file, 'w') as file:
        file.write(str(weekday_count))

def A4(filename="/data/contacts.json", targetfile="/data/contacts-sorted.json"):

    with open(filename, 'r') as file:
        contacts = json.load(file)

    
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

    
    with open(targetfile, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)

def A5(log_dir_path='/data/logs', output_file_path='/data/logs-recent.txt', num_files=10):
    log_dir = Path(log_dir_path)
    output_file = Path(output_file_path)

    
    log_files = sorted(log_dir.glob('*.log'), key=lambda f: f.stat().st_mtime, reverse=True)[:num_files]

    
    lines = []
    for log_file in log_files:
        with log_file.open('r', encoding='utf-8') as f_in:
            first_line = f_in.readline().strip()
            lines.append(first_line)

    
    with output_file.open('w', encoding='utf-8') as f_out:
        f_out.write("\n".join(lines) + "\n")


def A6(doc_dir_path='/data/docs', output_file_path='/data/docs/index.json'):
    docs_dir = doc_dir_path
    output_file = output_file_path
    index_data = {}

    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            
                            title = line[2:].strip()
                            
                            relative_path = os.path.relpath(file_path, docs_dir).replace('\\', '/')
                            index_data[relative_path] = title
                            break  
    
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)

import re

def A7(filename='/data/email.txt', output_file='/data/email-sender.txt'):
    
    with open(filename, 'r', encoding='utf-8') as file:
        email_content = file.read()

    
    match = re.search(r"(?i)^From:\s*.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?", email_content, re.MULTILINE)
    sender_email = match.group(1) if match else "Unknown"

    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(sender_email)

import base64
def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def A8(filename='/data/credit_card.txt', image_path='/data/credit_card.png'):
    
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "There is 8 or more digit number is there in this image, with space after every 4 digit, only extract the those digit number without spaces and return just the number without any other characters"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_to_base64(image_path)}"
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                             headers=headers, data=json.dumps(body))
    

    
    result = response.json()
    
    card_number = result['choices'][0]['message']['content'].replace(" ", "")

    
    with open(filename, 'w') as file:
        file.write(card_number)




import json
import requests
from scipy.spatial.distance import cosine

AIPROXY_TOKEN = "your_api_token_here"

def get_embeddings(texts):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    response = requests.post(
        "http://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers=headers,
        data=json.dumps(data)
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]

def A9(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):
    
    with open(filename, 'r') as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    if len(comments) < 2:
        print("Not enough comments to compare.")
        return

    
    embeddings = get_embeddings(comments)

    
    min_distance = float('inf')
    most_similar = None

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            distance = cosine(embeddings[i], embeddings[j])
            if distance < min_distance:
                min_distance = distance
                most_similar = (comments[i], comments[j])

    
    with open(output_filename, 'w') as f:
        f.write(most_similar[0] + '\n')
        f.write(most_similar[1] + '\n')

    print(f"Most similar comments written to {output_filename}")


def A10(filename='/data/ticket-sales.db', output_filename='/data/ticket-sales-gold.txt', query="SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"):
    
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    
    cursor.execute(query)
    total_sales = cursor.fetchone()[0]

    
    total_sales = total_sales if total_sales else 0

    
    with open(output_filename, 'w') as file:
        file.write(str(total_sales))

    conn.close()
