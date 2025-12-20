
```bash
curl https://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What do you think about AI coding assistants?",
    "max_tokens": 512,
    "temperature": 0.7
  }'
  ```