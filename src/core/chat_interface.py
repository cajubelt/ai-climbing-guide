from completion import get_completions_stream
from openai import OpenAI
import streamlit as st
from clients.elastic_client import ElasticClient

st.title("AI Climbing Guide")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
climbing_data_client = ElasticClient(elastic_url=st.secrets["ELASTICSEARCH_NODE_URL"], elastic_api_key=st.secrets["ELASTICSEARCH_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = get_completions_stream(openai_client, climbing_data_client, st.session_state["openai_model"], st.session_state.messages)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
