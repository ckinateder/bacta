# Examples

This directory contains examples of how to use the backtester.

## Setup

### Alpaca
These examples are set up to receive data from [Alpaca](https://alpaca.markets/). You will need to set up an Alpaca account and get an API key. Store the following variables in your environment:

```bash
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
DATA_DIR=data
```

Additionally, you will need to install the alpaca-py library. Run the following command:

```bash
pip install alpaca-py
```

### TA-Lib

You will also need to install the talib library. Follow [these instructions](https://github.com/TA-Lib/ta-lib-python) to install it for your system. For example, in a Debian-based system, you can run the following commands:

```bash
export TA_LIB_VERSION=0.6.4
wget https://github.com/ta-lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz
tar -xzf ta-lib-${TA_LIB_VERSION}-src.tar.gz
cd ta-lib-${TA_LIB_VERSION}
./configure --prefix=/usr
make
make install
```

Then, run the following command:

```bash
pip install ta-lib
```

## Usage

- `data.py`: A simple example of how to download data from Alpaca.
- `keltner_channel.ipynb`: A simple example of a backtester that uses the Keltner Channel to make decisions. Depends on TA-Lib.
- `ema_rsi.ipynb`: A simple EMA crossover strategy. Depends on TA-Lib.