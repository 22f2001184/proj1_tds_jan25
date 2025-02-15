# LLM-based Automation Agent

An automation agent that uses LLMs to process various tasks through an API interface.

## Setup

1. Clone the repository
2. Set up environment variables:
```bash
export AIPROXY_TOKEN="your_token_here"
```

3. Build the Docker image:
```bash
docker build -t llm-automation-agent .
```

4. Run the container:
```bash
docker run -p 8000:8000 -e AIPROXY_TOKEN=$AIPROXY_TOKEN llm-automation-agent
```

## API Endpoints

### POST /run
Execute a task:
```bash
curl -X POST "http://localhost:8000/run?task=your_task_description"
```

### GET /read
Read file contents:
```bash
curl "http://localhost:8000/read?path=/data/output.txt"
```

## Supported Tasks

### Phase A Tasks
- A1: Data generation setup
- A2: Markdown formatting
- A3: Date counting
- A4: Contact sorting
- A5: Log file processing
- A6: Markdown header extraction
- A7: Email sender extraction
- A8: Credit card OCR
- A9: Comment similarity analysis
- A10: Database operations

### Phase B Tasks
- API data fetching
- Git operations
- Database queries
- Web scraping
- Image processing
- Audio transcription
- Markdown to HTML conversion
- CSV/JSON API endpoints

## Security Features
- Restricted to /data directory
- No file deletion allowed
- Input validation
- Error handling