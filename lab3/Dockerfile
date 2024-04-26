FROM python:3.8

COPY . /app
WORKDIR /app

RUN python -m venv .venv
RUN . /app/.venv/bin/activate
RUN pip install -r requirements.txt

RUN . /app/model.sh

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
