FROM python:3.11

EXPOSE 8080
WORKDIR /app

COPY ./funes/ui/requirements.txt ./
COPY ./funes/ui/app.py ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]