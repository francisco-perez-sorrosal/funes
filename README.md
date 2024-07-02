




```
./tgi.sh
```


```
pixi run python -m streamlit run ./funes/ui/app.py --server.port=8502 \\n--server.address=0.0.0.0 \\n--server.enableCORS=false 
```

```
ssh -A -L 8601:localhost:8601 -L 8602:localhost:8602 -L  8000:localhost:8000 -L 8503:localhost:8603 fperez-gcloud-stupid-sailor-twift
```
