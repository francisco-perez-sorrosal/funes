# Server
```
./tgi.sh
```

# DSPy App
```
pixi run python -m streamlit run ./funes/ui/app.py --server.port=8502 \\n--server.address=0.0.0.0 \\n--server.enableCORS=false 
```

# Langgraph App
```
 pixi run python -m streamlit run ./funes/ui/base_app.py --server.port=8601 \\n--server.address=0.0.0.0 \\n--server.enableCORS=false  
```

# Tunnel
```
ssh -A -L 8601:localhost:8601 -L 8602:localhost:8602 -L 8501:localhost:8501 -L 8000:localhost:8000 fperez-gcloud-stupid-sailor-twift
```
