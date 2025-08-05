FROM python:3.13-bullseye

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