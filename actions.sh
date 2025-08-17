#! /bin/bash

# install dependencies
pip install -r requirements.txt

# run tests
pytest tests/

# build
python -m build