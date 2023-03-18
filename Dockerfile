FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY ./scripts/download_models.sh /app/download_models.sh

RUN /app/download_models.sh

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]