import streamlit as st
import os, wikipedia, requests, time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_functions,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.document_loaders import NewsURLLoader, BraveSearchLoader


load_dotenv()


####### -------------------------------------- 1. CREATE THE TOOLS ----------------------------------------

### GET HEADLINES TOOL


class CountryCodeInput(BaseModel):
    countrycode: str = Field(
        description="Two-letter ISO 3166-1 country code to search headlines for."
    )


@tool(args_schema=CountryCodeInput)
def get_headlines(countrycode):
    """Gets webpage links of latest headlines about a country. Use this tool when user cites the name of a country. Use 'web_retriever' to load the webpage links to read the content."""
    BASE_URL = "https://newsapi.org/v2/top-headlines?"

    try:
        params = {
            "apiKey": st.secrets["newsapi_api_key"],
            "country": countrycode,
            "pageSize": 5,
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
    """Gets webpage links of news about a personality, issue or event. Use this tool when user asks for the latest news or headlines about a personality, issue or event. Use 'web_retriever' to load the webpage links to read the content."""
    BASE_URL = "https://newsapi.org/v2/everything?"
    try:
        params = {
            "apiKey": st.secrets["newsapi_api_key"],
            "q": query,
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


### GET ANSWERS TOOL
class SearchInput(BaseModel):
    query: str = Field(..., description="query to search for")


@tool(args_schema=SearchInput)
def answer_search(query):
    """Search the internet for answers based on the query. Use this tool when user asks a question about a personality, issue or event. Use 'web_retriever' to load the webpage links to read the content."""
    try:
        loader = BraveSearchLoader(
            query=query,
            api_key=st.secrets["brave_api_key"],
            search_kwargs={"count": 5},
        )
        docs = loader.load()
        return docs
    except Exception as e:
        return f"An error has occurred: {e}"


### WEBPAGE RETRIEVER TOOL
class UrlListInput(BaseModel):
    url_list: List[str] = Field(..., description="List of url links to web pages")


@tool(args_schema=UrlListInput)
def webpage_retriever(url_list):
    """Use this to load and read the news websites from the 'get_news' and 'answer_search' tools"""
    summaries = []
    model = ChatOpenAI(openai_api_key=st.secrets["openai_api_key"])
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
    """Run Wikipedia search and get page summaries. Only use this tool as a last resort if there are no news results available."""
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


tools = [get_news, answer_search, webpage_retriever]

###### ------------------------------------ 2. CREATE THE CHAIN ---------------------------------------------
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
            """You are a friendly chat assistant who can provide the latest news and/or answers about personalities, issues, events. Look for news only if the user asks for headlines or news.
            
            If the search doesn't return any results, try varying the search terms or splitting them up.
            
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
    """    Ask me for the latest news about a personality, a company, an issue, an event etc. Feel free to ask questions too."""
)
prompt = st.chat_input("Say something")

if prompt:
    st.write(f"🔍🔍 Looking for: {prompt}...")

    ### Simple execution
    # result = agent_executor.invoke({"input": prompt})
    # st.write(result["output"])

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

    # ### Streaming only the tool used
    for chunk in agent_executor.stream({"input": prompt}):
        if "actions" in chunk:
            if chunk["actions"][0].tool == "answer_search":
                st.write("👨🏻‍💻👨🏻‍💻👨🏻‍💻 Searching the web...")
            elif chunk["actions"][0].tool == "get_news":
                st.write("🗞️🗞️🗞️ Getting the latest news...")
            elif chunk["actions"][0].tool == "webpage_retriever":
                st.write("👾👾👾 Retrieving info...")
            else:
                st.write("⌛️⌛️⌛️ Just a moment more...")
        elif "steps" in chunk:
            st.write("🕵🏻🕵🏻🕵🏻 Analysing results...")
        elif "output" in chunk:
            st.write(f"😽😽😽 Here you go!\n\n{chunk['output']}")
        else:
            raise ValueError()
