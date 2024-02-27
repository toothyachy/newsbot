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

# Tools
from utils.tools import get_news, answer_search, country_news_search, webpage_retriever

tools = [get_news, answer_search, country_news_search, webpage_retriever]

# Prompt templates
from templates.prompts import newsbot_prompt


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
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | prompt
    | model_functions
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)
