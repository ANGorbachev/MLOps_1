FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "translator.py", "--server.port=8501", "--server.address=0.0.0.0"]
