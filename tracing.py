from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

from dotenv import load_dotenv

load_dotenv()


prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()


print(chain.invoke({"topic": "LangSmith"}))
