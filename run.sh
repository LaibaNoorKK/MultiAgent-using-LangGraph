#python -m streamlit run main.py --server.port 8000 --server.address 0.0.0.0
#!/bin/bash
gunicorn auth_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
