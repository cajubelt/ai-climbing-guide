# AI Climbing Guide

## Motivation

When I go on climbing trips with friends, there is often a fairly intensive optimization problem the group needs to solve to make sure everyone has fun-- finding co-located climbs that meet everyone's ability levels and stylistic preferences. Even for small groups of 2 or 3, this can take hours, and eats into time most people would prefer to spend hanging out with their friends, rather than poring through guidebooks.

With this project my goal is to develop a RAG-enabled AI agent that can answer the following prompt (or similar ones) well:
```
I’m going climbing with my friends Eli and Kristi. Eli climbs around v7 and prefers crimps. He also prefers a safe landing zone and no highballs. Kristi climbs around v6. One climb she really loved was The Pearl. I climb around v9, and I enjoy a range of styles, just not pockets. We’re going to Tahoe for the weekend and we’ll be staying in Truckee. Can you recommend some bouldering spots for us where we’ll all have a good time?
```

When I tried this prompt with vanilla ChatGPT 4o, it gives a really great answer, except when you look up the climbs it suggest-- which either don't exist, or are hundreds of miles away. Not very helpful!

## Implementation Status

I've built out some but not all of the functionality I'd hope for from an AI climbing guide. Here are some screenshots demonstrating the present-day capabilities of the app:
<img width="757" alt="Screenshot 2025-02-04 at 5 16 41 PM" src="https://github.com/user-attachments/assets/9bf292c7-6634-4ffc-ac1b-ac3e8aae4b35" />
<img width="727" alt="Screenshot 2025-02-04 at 5 16 28 PM" src="https://github.com/user-attachments/assets/42e44574-171d-438f-8f51-20a691309920" />
<img width="761" alt="Screenshot 2025-02-04 at 4 55 25 PM" src="https://github.com/user-attachments/assets/f7ff2a5a-3359-4b09-a796-f821106caf0e" />

- [x] find or develop a sizeable open source climbing database
- [x] write a script that loads climbing data into a database
- [x] write a search API that queries the db with basic searches
- [x] build a basic chat UI
- [x] leverage OpenAI function calling API to enable AI-driven climbing data db lookups
- [ ] power up the AI enough to get a great answer for the prompt from [Motivation](#motivation) (or a similar one with areas included in the climb db). This may require [CoRAG](https://arxiv.org/abs/2501.14342) or a more powerful search API, TBD
- [ ] account creation and auth via login with Google
- [ ] limit messages for free tier users
- [ ] deploy!

## Contents

This repo consists of:
- a script for loading OpenBeta climbing data into an ElasticSearch index
- a Streamlit interface for talking to an AI Climbing Guide

## Setup

### OpenBeta ETL

In order to populate an ElasticSearch index with OpenBeta climbs, you must first do the following.

1. Install Python
1. Install Pipenv
1. Install dependencies
1. Create an ElasticSearch deployment. This can be done locally in [one terminal command](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html)
1. Get an OpenAI API key
1. Copy `.env.example` to `.env` and fill in the secrets

You can now run the script with `PYTHONPATH=./src pipenv run python ./scripts/load_climbing_data.py`

### StreamLit Interface

1. Populate an ElasticSearch index as described above
1. Fill in the same secrets as in the `.env` file in `.streamlit/secrets.toml`
1. Run `PYTHONPATH=./src pipenv run python -m streamlit run ./src/core/chat_interface.py`

A URL should print to the console for your AI Climbing Guide app.
