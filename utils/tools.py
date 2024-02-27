import wikipedia, requests
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import NewsURLLoader, BraveSearchLoader
import streamlit as st

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


### GET COUNTRY NEWS TOOL
class CountrySearchInput(BaseModel):
    query: str = Field(..., description="country to search news for e.g. Singapore")


@tool(args_schema=CountrySearchInput)
def country_news_search(query):
    """Search the internet for latest news about a country. Use 'web_retriever' to load the webpage links to read the content."""
    try:
        BASE_URL = "https://api.search.brave.com/res/v1/news/search"
        params = {
            "q": query,
            "count": 5,
            "search_lang": "en",
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": st.secrets["brave_api_key"],
        }
        response = requests.get(BASE_URL, params=params, headers=headers)
        results = response.json()["results"]
        news = []
        for r in results:
            news.append(
                {"title": r["title"], "url": r["url"], "description": r["description"]}
            )
        return news
    except Exception as e:
        return f"An error has occurred: {e}"


### WEBPAGE RETRIEVER TOOL
class UrlListInput(BaseModel):
    url_list: List[str] = Field(..., description="List of url links to web pages")


@tool(args_schema=UrlListInput)
def webpage_retriever(url_list):
    """Use this to load and read the news websites from the 'get_news', 'answer_search' and 'country_news_search' tools"""
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
