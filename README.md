# Bacta

Bacta is a Python library for backtesting trading strategies.
![symbols](img/performance_analysis.png)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install bacta.

```bash
pip install bacta
```

## Usage

### Making Plots

## Development

A docker image is provided for convenience. To build the image, run the following command:

```bash
docker build -t bacta . 
```

To run the image, run the following command:

```bash
docker run -it bacta --env-file .env
```

### Testing

To run the tests, run the following command:

```bash
python -m unittest discover tests
```

### Uploading to PyPI

[To upload the package to PyPI](https://packaging.python.org/en/latest/tutorials/packaging-projects/), run the following commands to build and upload the package. You will need to have a [PyPI account](https://pypi.org/account/register/).

```bash
python -m build
twine upload dist/*
```


## License

[MIT](https://choosealicense.com/licenses/mit/)