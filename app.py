# /// srcipt
# requires-python = ">=3.12"
# dependencies = [
#       "fastapi", 
#       "uvicorn", 
#       "requests"
# ]
# ///


from urllib import response
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


tools = [  # Changed from { to [
    {
        "type": "function",
        "function": {
            "name": "script_runner",
            "description": "Install a package and run a script from a url with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of arguments to pass to the script."
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    }
]


AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDExODRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ob_Orv5l-b3jsbcKwsjwI99_r9CjKTJ1oK829zpZS4g"
    # Remove the raise statement since we're providing a default token

@app.get("/")
def home ():
    return {"Yay TDS is awesome!"}

@app.get("/read")
def read_file(path: str):
    try :
        with open(path, "r") as f:
            return {"data": f.read()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/run")
def task_runner(task:str):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"  # Fixed f-string syntax
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": task
            },
            {
                "role": "system",
                "content": """
You are an assistant who has to do a variety of tasks.
If your task involves running a script, you can use the script_runner tool.
If your task involves writing a code, you can use the task_runner tool.
                """
            }
        ],
        "tools": tools,
        "tool_choice": "auto"  # Fixed key name from "tools:choices" to "tool_choice"
    }
    response = requests.post(url=url, headers=headers, json=data)  # Fixed requests.post and variable name
    return response.json()

















if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)