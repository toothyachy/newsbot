import streamlit as st
import os, wikipedia, requests, time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import (
    tool,
    AgentExecutor,
    create_openai_functions_agent,
    load_tools,
)
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_functions,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_core.agents import AgentFinish
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_community.document_loaders import NewsURLLoader

load_dotenv()


####### -------------------------------------- 1. CREATE THE TOOLS ----------------------------------------

### GET HEADLINES TOOL


class CountryCodeInput(BaseModel):
    countrycode: str = Field(
        description="Two-letter ISO 3166-1 country code to search headlines for."
    )


@tool(args_schema=CountryCodeInput)
def get_headlines(countrycode):
    """Gets webpage links of latest headlines about a country. Use 'web_retriever' to load the webpage links to read the content."""
    BASE_URL = "https://newsapi.org/v2/top-headlines?"

    try:
        params = {
            "apiKey": os.environ["NEWSAPI_API_KEY"],
            "country": countrycode,
            "pageSize": 10,
        }

        response = requests.get(BASE_URL, params)
        response = response.json()
        news = []
        if response["totalResults"] == 0:
            return "No headlines found"
        else:
            news = [i["url"] for i in response["articles"]]
            return news

    except Exception as e:
        return f"An error has occurred: {e}"


### GET NEWS TOOL
class NewsInput(BaseModel):
    query: str = Field(..., description="Query to search the news for")


@tool(args_schema=NewsInput)
def get_news(query):
    """Gets webpage links of news about a personality or event. When given the name of a personality or event, always use this tool first before trying 'wikipedia_search'. Use 'web_retriever' to load the webpage links to read the content."""
    BASE_URL = "https://newsapi.org/v2/everything?"
    try:
        params = {
            "apiKey": os.environ["NEWSAPI_API_KEY"],
            "q": query,
            # "sources": news_sources,
            "pageSize": 5,
        }

        response = requests.get(BASE_URL, params)
        response = response.json()
        news = []
        if response["totalResults"] == 0:
            return "No latest news found"
        else:
            news = [i["url"] for i in response["articles"]]
            return news

    except Exception as e:
        return f"An error has occurred: {e}"


### WEBPAGE RETRIEVER TOOL
# Define the class
class UrlListInput(BaseModel):
    url_list: List[str] = Field(..., description="List of url links to web pages")


@tool(args_schema=UrlListInput)
def webpage_retriever(url_list):
    """Use this to load and read the news websites from the 'country_news_search' and 'news_search' tools"""
    summaries = []
    model = ChatOpenAI()
    try:
        # Load website
        loader = NewsURLLoader(url_list)
        docs = loader.load()  # list
        for i in docs:
            summary = model.invoke(f"Summarise {i}")
            summaries.append(summary)
        return summaries
    except Exception as e:
        return f"An error has occurred: {e}"


### WIKIPEDIA TOOL
class WikiInput(BaseModel):
    query: str = Field(..., description="Query to search Wikipedia for")


@tool(args_schema=WikiInput)
def wikipedia_search(query):
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)  # Returns a list of page titles
    if page_titles:
        print(f"No of page titles: {len(page_titles)}. The list has {page_titles}")
    summaries = []
    for page_title in page_titles[:3]:
        try:
            print(f"now working on page title: {page_title}")
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia search results found"
    summaries_str = "\n\n".join(summaries)
    return summaries_str


tools = [get_headlines, get_news, webpage_retriever, wikipedia_search]

###### ------------------------------------ 2. CREATE THE CHAIN ---------------------------------------------
model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
functions = [convert_to_openai_function(f) for f in tools]
model_functions = model.bind(functions=functions)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly chat assistant who checks the latest news and facts about personalities, events and countries when asked. If there are no or few news results, please search wikipedia. If the search doesn't return any results, try varying the search terms or splitting them up.
            
            In your reply to the user:
            1. Think carefully then provide a detailed summary of the information you have received. 
            2. Include at the end the list of webpages you analysed and their corresponding url links
            """,
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


###### ------------------------------------- 3. STREAMLIT IT --------------------------------------------
st.title("🤖 News GPT")
st.info(
    """    Ask me for the latest news about a country, personality, company or event. All you need to do is enter a country, a name or an event, I'll extract the latest news if there's any; if none, you'll just get wiki info."""
)
prompt = st.chat_input("Say something")

if prompt:
    st.write(f"🔍🔍 Looking for: {prompt}...")

    ### Simple execution
    result = agent_executor.invoke({"input": prompt})
    st.write(result["output"])

    ### Streaming all message logs
    # for chunk in agent_executor.stream({"input": prompt}):
    #     st.write(chunk)

    # ### Streaming only key developments
    # for chunk in agent_executor.stream({"input": prompt}):
    #     if "actions" in chunk:
    #         st.write(
    #             f"Tool: {chunk['actions'][0].tool}. Tool input: {chunk['actions'][0].tool_input}"
    #         )
    #     elif "steps" in chunk:
    #         st.write(f"Tool result: {chunk['steps'][0].observation}.")
    #     elif "output" in chunk:
    #         st.write(f"{chunk['output']}")
    #     else:
    #         raise ValueError()
