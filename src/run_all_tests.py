import pytest
from langchain.document_loaders import YoutubeLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from main import *

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def run_all_tests():
    pytest.main(["-v", "test_files.py"])


if __name__ == "__main__":
    run_all_tests()
