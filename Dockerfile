FROM python:3.10
#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

#CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0"]
CMD ["uvicorn", "src.app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
