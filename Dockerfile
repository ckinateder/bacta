FROM python:3.13.3-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential git

WORKDIR /app

COPY requirements.txt .
COPY .env .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]