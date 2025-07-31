FROM python:3.13.3-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential git wget

# cd to app directory
WORKDIR /app

# copy requirements and .env
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

CMD ["/bin/bash"]