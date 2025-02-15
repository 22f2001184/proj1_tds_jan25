# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "requests",
#   "Pillow",
#   "beautifulsoup4",
#   "markdown",
#   "SpeechRecognition",
#   "opencv-python",
#   "numpy",
#   "sentence-transformers",
#   "scikit-learn",
#   "python-multipart",
#   "python-dotenv"
# ]
# ///

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import subprocess
import glob
from typing import Dict, Any
import base64
from PIL import Image
import io
from bs4 import BeautifulSoup
import markdown
import speech_recognition as sr
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_TOKEN = os.environ.get("AIPROXY_TOKEN")
LLM_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Task Functions
def count_weekday(input_file: str, output_file: str, weekday: int):
    """Count occurrences of specific weekday in dates file"""
    with open(input_file, 'r') as f:
        dates = f.readlines()
    count = sum(1 for date in dates if datetime.strptime(date.strip(), '%Y-%m-%d').weekday() == weekday)
    with open(output_file, 'w') as f:
        f.write(str(count))
    return {"status": "success", "count": count}

def count_wednesdays(input_file: str, output_file: str) -> dict:
    """
    Count Wednesdays in a dates file with mixed date formats.
    Handles formats like:
    - YYYY-MM-DD
    - DD-Mon-YYYY
    - Mon DD, YYYY
    - YYYY/MM/DD HH:MM:SS
    """
    def parse_date(date_str: str) -> datetime:
        date_str = date_str.strip()
        formats = [
            '%Y-%m-%d',           # 2008-09-29
            '%d-%b-%Y',           # 29-Apr-2003
            '%b %d, %Y',          # Feb 19, 2001
            '%Y/%m/%d %H:%M:%S'   # 2017/07/10 19:44:26
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    wednesday_count = 0
    with open(input_file, 'r') as f:
        for line in f:
            date = parse_date(line)
            if date and date.weekday() == 2:  # 2 = Wednesday
                wednesday_count += 1

    with open(output_file, 'w') as f:
        f.write(str(wednesday_count))
    
    return {"status": "success", "count": wednesday_count}

def sort_contacts(input_file: str, output_file: str):
    """Sort contacts by last_name, first_name"""
    with open(input_file, 'r') as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
    with open(output_file, 'w') as f:
        json.dump(sorted_contacts, f)
    return {"status": "success"}

def format_markdown(file_path: str, prettier_version: str = "3.4.2"):
    """Format markdown using prettier"""
    # Security check
    if not file_path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid file path - Must be in /data directory")
    
    try:
        # Check if npm and npx are available
        npm_path = subprocess.run(["which", "npm"], capture_output=True, text=True).stdout.strip()
        if not npm_path:
            raise HTTPException(status_code=500, detail="npm is not installed")
        
        npx_path = subprocess.run(["which", "npx"], capture_output=True, text=True).stdout.strip()
        if not npx_path:
            # Try to find npx in npm directory
            npx_path = "/usr/local/bin/npx"
            if not os.path.exists(npx_path):
                raise HTTPException(status_code=500, detail="npx is not installed")
        
        # Install prettier if needed
        subprocess.run([npx_path, "--yes", f"prettier@{prettier_version}"], 
                      check=True, 
                      capture_output=True)
        
        # Format file in-place
        result = subprocess.run(
            [
                npx_path,
                f"prettier@{prettier_version}",
                "--write",
                "--parser", "markdown",
                "--prose-wrap", "always",
                "--print-width", "80",
                file_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {"status": "success", "message": "File formatted successfully"}
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Prettier failed: {result.stderr}"
            )
            
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required dependency not found: {str(e)}"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prettier execution failed: {str(e)}"
        )

def extract_log_lines(log_dir: str, output_file: str, num_files: int = 10):
    """Extract first lines from recent log files"""
    log_files = sorted(glob.glob(f'{log_dir}/*.log'), key=os.path.getmtime, reverse=True)[:num_files]
    with open(output_file, 'w') as outfile:
        for log in log_files:
            with open(log, 'r') as infile:
                first_line = infile.readline().strip()
                outfile.write(f"{first_line}\n")
    return {"status": "success"}

def extract_email_sender(input_file: str, output_file: str) -> dict:
    """Extract sender's email address from email content using LLM"""
    # Security check
    if not input_file.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid file path - Must be in /data directory")
    
    try:
        # Read email content
        with open(input_file, 'r') as f:
            email_content = f.read()
        
        # Query LLM to extract email
        response = requests.post(
            LLM_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_TOKEN}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Extract ONLY the sender's email address from this email. Return ONLY the email address, nothing else:\n\n{email_content}"
                    }
                ]
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM API request failed")
        
        # Extract email from response
        email_address = response.json()['choices'][0]['message']['content'].strip()
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(email_address)
        
        return {"status": "success", "email": email_address}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Email file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def query_llm(prompt: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(LLM_API_URL, headers=headers, json=data)
    return response.json()

# Additional Task Functions
def setup_data_generator(email: str):
    """Install uv and run data generator"""
    try:
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        response = requests.get(url)
        with open("datagen.py", "w") as f:
            f.write(response.text)
        subprocess.run(["python", "datagen.py", email])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_markdown_headers(docs_dir: str, output_file: str):
    """Extract H1 headers from markdown files"""
    index = {}
    for md_file in Path(docs_dir).glob("*.md"):
        with open(md_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if (line.startswith('# ')):
                    index[md_file.name] = line[2:].strip()
                    break
    with open(output_file, 'w') as f:
        json.dump(index, f)
    return {"status": "success"}

def extract_card_number(image_path: str, output_file: str):
    """Extract credit card number using OCR and LLM"""
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    prompt = f"Extract the credit card number from this image (base64): {img_data}"
    response = query_llm(prompt)
    card_number = ''.join(filter(str.isdigit, response['choices'][0]['message']['content']))
    
    with open(output_file, 'w') as f:
        f.write(card_number)
    return {"status": "success"}

def find_similar_comments(input_file: str, output_file: str) -> dict:
    """Find the most similar pair of comments using sentence embeddings"""
    # Security check
    if not input_file.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid file path - Must be in /data directory")
    
    try:
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Read comments
        with open(input_file, 'r') as f:
            comments = [line.strip() for line in f.readlines()]
        
        if len(comments) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 comments to compare")
        
        # Generate embeddings
        embeddings = model.encode(comments)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Set diagonal to -1 to ignore self-similarity
        np.fill_diagonal(similarity_matrix, -1)
        
        # Find most similar pair
        max_i, max_j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        
        # Write the similar pair to output file
        with open(output_file, 'w') as f:
            f.write(f"{comments[max_i]}\n{comments[max_j]}")
        
        return {
            "status": "success",
            "similarity_score": float(similarity_matrix[max_i, max_j]),
            "comment1": comments[max_i],
            "comment2": comments[max_j]
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Comments file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_ticket_sales(db_path: str, output_file: str):
    """Calculate total sales for Gold tickets"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    total = cursor.fetchone()[0]
    conn.close()
    
    with open(output_file, 'w') as f:
        f.write(str(total))
    return {"status": "success"}

def calculate_ticket_sales(input_db: str, output_file: str) -> dict:
    """Calculate total sales for Gold tickets from SQLite database"""
    # Security check
    if not input_db.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid file path - Must be in /data directory")
    
    try:
        # Connect to database
        conn = sqlite3.connect(input_db)
        cursor = conn.cursor()
        
        # Calculate total sales for Gold tickets
        cursor.execute("""
            SELECT COALESCE(SUM(units * price), 0) 
            FROM tickets 
            WHERE type = 'Gold'
        """)
        
        total_sales = cursor.fetchone()[0]
        
        # Close database connection
        conn.close()
        
        # Write result to output file
        with open(output_file, 'w') as f:
            f.write(str(total_sales))
        
        return {
            "status": "success",
            "total_sales": total_sales
        }
        
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Business Task Functions
def fetch_api_data(api_url: str, output_file: str):
    """Fetch data from API and save it"""
    response = requests.get(api_url)
    with open(output_file, 'w') as f:
        json.dump(response.json(), f)
    return {"status": "success"}

def git_operations(repo_url: str, commit_message: str):
    """Clone repo and make commit"""
    subprocess.run(["git", "clone", repo_url])
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    subprocess.run(["git", "add", "."], cwd=repo_name)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_name)
    return {"status": "success"}

def scrape_website(url: str, output_file: str):
    """Extract data from website"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    with open(output_file, 'w') as f:
        json.dump({"content": soup.get_text()}, f)
    return {"status": "success"}

def process_image(image_path: str, output_path: str, max_size: int):
    """Compress or resize image"""
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        img.save(output_path, optimize=True)
    return {"status": "success"}

def transcribe_audio(audio_path: str, output_file: str):
    """Transcribe MP3 to text"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        with open(output_file, 'w') as f:
            f.write(text)
    return {"status": "success"}

def convert_markdown(input_file: str, output_file: str):
    """Convert Markdown to HTML"""
    with open(input_file, 'r') as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    with open(output_file, 'w') as f:
        f.write(html_content)
    return {"status": "success"}

# Function Definitions for Tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_weekday",
            "description": "Count occurrences of specific weekday in a dates file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "weekday": {"type": "integer", "description": "0=Monday, 6=Sunday"}
                },
                "required": ["input_file", "output_file", "weekday"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts by last_name and first_name",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string"},
                    "output_file": {"type": "string"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "setup_data_generator",
            "description": "Setup and run data generator",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_markdown_headers",
            "description": "Extract H1 headers from markdown files",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {"type": "string"},
                    "output_file": {"type": "string"}
                },
                "required": ["docs_dir", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_markdown",
            "description": "Format a markdown file using prettier",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the markdown file to format (must be in /data directory)"
                    },
                    "prettier_version": {
                        "type": "string",
                        "description": "Prettier version to use",
                        "default": "3.4.2"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_wednesdays",
            "description": "Count number of Wednesdays in a dates file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input file containing dates"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file where count will be written"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract sender's email address from email content using LLM",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input file containing email content"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file where email address will be written"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "description": "Find the most similar pair of comments using embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input file containing comments"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file where similar comments will be written"
                    }
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ticket_sales",
            "description": "Calculate total sales for Gold tickets from SQLite database",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_db": {
                        "type": "string",
                        "description": "Path to SQLite database file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file where total sales will be written"
                    }
                },
                "required": ["input_db", "output_file"]
            }
        }
    }
]

def query_gpt(user_input: str) -> Dict[str, Any]:
    """Query GPT with tools for function calling"""
    response = requests.post(
        LLM_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a task execution assistant. Process tasks and call appropriate functions."},
                {"role": "user", "content": user_input}
            ],
            "tools": TOOLS,
            "tool_choice": "auto"
        }
    )
    return response.json()

@app.post("/run")
async def run(task: str):
    # Security check
    if "../" in task or task.startswith("/") or "delete" in task.lower():
        raise HTTPException(status_code=400, detail="Invalid task - Security violation")
    
    try:
        response = query_gpt(task)
        if 'choices' in response and len(response['choices']) > 0:
            tool_call = response['choices'][0]['message'].get('tool_calls', [])
            if tool_call:
                function_name = tool_call[0]['function']['name']
                arguments = json.loads(tool_call[0]['function']['arguments'])
                func = globals()[function_name]
                return func(**arguments)
        raise HTTPException(status_code=400, detail="Could not process task")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read(path: str):
    if "../" in path or not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404)

def execute_task(task: str):
    # ...existing security checks...
    
    try:
        if "comments.txt" in task and "similar" in task.lower():
            return find_similar_comments("/data/comments.txt", "/data/comments-similar.txt")
        if "email.txt" in task and ("sender" in task.lower() or "email address" in task.lower()):
            return extract_email_sender("/data/email.txt", "/data/email-sender.txt")
        if "ticket-sales.db" in task and "gold" in task.lower():
            return calculate_ticket_sales("/data/ticket-sales.db", "/data/ticket-sales-gold.txt")
        # ...existing task handlers...
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)