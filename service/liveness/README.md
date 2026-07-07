# Liveness Detection Demo test

// Abrir el server
.\.venv\Scripts\python.exe -m uvicorn serve_liveness:app --host 127.0.0.1 --port 8000

// Iniciar el cliente en el navegador
http://localhost:8000/v1/liveness/client
