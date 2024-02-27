import streamlit as st
from agent import (
    agent_with_message_history,
    get_session_history,
    store,
    current_session_id,
    reset_session,
)

# Enable LangSmith tracing
import os

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["langchain_tracing_v2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["langchain_project"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["langchain_endpoint"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_key"]


st.title("🤖 News GPT")
st.caption(
    """    Chat with me about the latest news or facts about a country, a personality, a company, an issue or an event!"""
)

# Initiate/get session history with current session id
get_session_history(session_id=current_session_id)
session_limit = 8

# # Display chat messages from history on app rerun
history = store[current_session_id].messages
for i in range(len(history)):
    if i % 2 == 0:
        with st.chat_message("user"):
            st.write(f"{history[i].content}")
    else:
        with st.chat_message("assistant"):
            st.write(f"{history[i].content}")


if qn := st.chat_input("Ask away!"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(qn)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = agent_with_message_history.stream(
            {"input": qn},
            config={"configurable": {"session_id": current_session_id}},
        )
        for chunk in stream:
            if "actions" in chunk:
                if "query" in chunk["actions"][0].tool_input:
                    st.write(f"Looking for '{chunk['actions'][0].tool_input['query']}'")
                elif chunk["actions"][0].tool == "answer_search":
                    st.write("👨🏻‍💻👨🏻‍💻 Searching the web...")
                elif chunk["actions"][0].tool == "get_news":
                    st.write("🗞️🗞️ Getting the latest news...")
                elif chunk["actions"][0].tool == "country_news_search":
                    st.write("🌎🌎 Getting country news...")
                elif chunk["actions"][0].tool == "webpage_retriever":
                    st.write("👾👾 Retrieving info...")
                else:
                    st.write("⌛️⌛️ Just a moment more...")
            elif "steps" in chunk:
                st.write("😽😽 Analysing results...")
            elif "output" in chunk:
                st.write(f"{chunk['output']}")
            else:
                raise ValueError()

    # If session limit is reached, delete current session key-value pair from store and generate new current session id
    if len(history) >= session_limit:
        st.warning("Resetting session state", icon="⚠️")
        reset_session(current_session_id)

print(len(history), current_session_id)
