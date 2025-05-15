# Arby's

Prototyping statistical arbitrage.

![tickers](img/Utility-Spread.png)

## Stack
- Python
- SimFin
- Alpaca

## Usage

### Development

My `.env` file has the following fields:

```yaml
SIMFIN_API_KEY=<api key here>
SIMFIN_DATA_DIR=data/simfin
DATA_DIR=data
```

#### Docker Setup

Run the following command to build the docker image:

```bash
docker build -t arbys . 
```

Run the following command to run the docker image:

```bash
docker run -it --rm  -v $(pwd):/app -w /app arbys
```
**Note that you may have issues showing `matplotlib` images when inside the docker container; a workaround is to [just save the generated plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) instead.**