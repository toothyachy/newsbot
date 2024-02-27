# OpenAI model
from langchain_openai import ChatOpenAI

# Prompts
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Agents and methods
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_functions,
)

# Schemas
from langchain.schema.runnable import RunnablePassthrough

# Function calling
from langchain_core.utils.function_calling import convert_to_openai_function

# Memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Streamlit
import streamlit as st

# Others
import random

# Custom tools and prompts
from templates.prompts import newsbot_prompt
from utils.tools import answer_search, news_search, webpage_retriever

tools = [answer_search, news_search, webpage_retriever]


# # MODEL AND PROMPT
model = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0,
    openai_api_key=st.secrets["openai_api_key"],
)
functions = [convert_to_openai_function(f) for f in tools]
model_functions = model.bind(functions=functions)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            newsbot_prompt,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# # BASIC CHAIN AND AGENT

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | prompt
    | model_functions
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

# # MEMORY MANAGEMENT
store = st.session_state


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # adds a new entry to the `store` dictionary with the session ID as the key and an instance of `ChatMessageHistory` as the value, i.e. a tuple containing an attribute 'messages' which is an empty list of messages.

    return store[session_id]


def gen_session_id():
    session_id = random.randint(100000, 999999)
    return str(session_id)


def reset_session(current_session_id):
    del st.session_state[current_session_id]
    current_session_id = gen_session_id()
    return current_session_id


# # Initialise session
current_session_id = gen_session_id


# # AGENT WITH MEMORY
agent_with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
