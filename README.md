# AI Climbing Guide

This repo consists of:
- a script for loading OpenBeta climbing data into an ElasticSearch index
- a Streamlit interface for talking to an AI Climbing Guide

## Setup

### OpenBeta ETL

In order to populate an ElasticSearch index with OpenBeta climbs, you must first do the following.

1. Install Python
1. Install Pipenv
1. Install dependencies
1. Create an ElasticSearch index
1. Get an OpenAI API key
1. Create a `.env` file with secrets

You can now run the script with `PYTHONPATH=./src pipenv run python ./scripts/load_climbing_data.py`

### StreamLit Interface

1. Populate an ElasticSearch index as described above
1. Run `PYTHONPATH=./src pipenv run python -m streamlit run ./src/core/chat_interface.py`

A URL should print to the console for your AI Climbing Guide app.
