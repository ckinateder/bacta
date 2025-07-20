FROM python:3.13.3-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential git wget

# install TA-lib
ENV TA_LIB_VERSION=0.6.4
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz
RUN tar -xzf ta-lib-${TA_LIB_VERSION}-src.tar.gz
RUN cd ta-lib-${TA_LIB_VERSION} && ./configure --prefix=/usr && make && make install
RUN ldconfig

# cd to app directory
WORKDIR /app

# copy requirements and .env
COPY requirements.txt .
COPY .env .

# install dependencies
RUN pip install -r requirements.txt

CMD ["/bin/bash"]