FROM ubuntu:noble
# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    zlib1g-dev libffi-dev libssl-dev \
    libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev tk-dev \
    libncursesw5-dev libgdbm-dev uuid-dev libdb-dev \   
    unzip 

# uninstall current python
RUN apt-get remove -y python3 python3-pip

# Install Python 3.13 from source
WORKDIR /opt
ENV PYTHON_VERSION=3.13.3
RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}*

# Ensure `python` and `pip3` point to Python 3.1
RUN ln -sf /usr/local/bin/python3.13 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.13 /usr/bin/pip

# Install Node.js (required by Kaleido)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs


# set build arg for ta-lib
ARG WITH_EXTRAS=false

# install ta-lib if WITH_TA_LIB is true
RUN if [ "$WITH_EXTRAS" = "true" ]; then \
    export TA_LIB_VERSION=0.6.4; \
    wget https://github.com/ta-lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz && \
    tar -xzf ta-lib-${TA_LIB_VERSION}-src.tar.gz && \
    cd ta-lib-${TA_LIB_VERSION} && \
    ./configure --prefix=/usr && \
    make && \
    make install; \
    fi

# cd to app directory
WORKDIR /app

# copy requirements
COPY requirements.txt .
COPY requirements-extras.txt .

RUN if [ "$WITH_EXTRAS" = "true" ]; then \
    pip install -r requirements-extras.txt; \
    else \
    pip install -r requirements.txt; \
    fi

CMD ["/bin/bash"]