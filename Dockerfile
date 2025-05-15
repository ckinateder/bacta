FROM python:3.13.3-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY .env .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]