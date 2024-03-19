import sys
import os
from typing import Union, List
from urllib.parse import urlparse
from cmdline import Switch, parse

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs


def extract_youtube_link_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            return content
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def is_youtube_link(input_str):
    # Check if the input string is a valid YouTube link
    parsed_url = urlparse(input_str)
    return (
            parsed_url.netloc == "www.youtube.com"
            and "/watch" in parsed_url.path
            and "v=" in parsed_url.query
    )


def create_db_from_powerpoint_file(pptx_file):
    loader = UnstructuredPowerPointLoader(pptx_file)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs


def create_db_from_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-4", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that provides metadata on pdfs regarding psychology research surveys
         using this transcript: {docs}.

        Every question in the survey will be categorized under a psychological facet. Make sure to properly
        categorize each question with the proper facet, usually given by the subheader, and assign each facet a number.
        If another question appears with the same facet, use the same number. 
        Make sure to sort the csv in ascending order by facet_num.
        

        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    return response, docs


def process_file(file_path):
    _, ext = os.path.splitext(file_path)

    if ext == ".pptx":
        db, docs = create_db_from_powerpoint_file(file_path)
    elif ext == ".pdf":
        db, docs = create_db_from_pdf(file_path)
    elif ext == '.txt':
        input_str = extract_youtube_link_from_file(file_path)
        if is_youtube_link(input_str):
            try:
                db, docs = create_db_from_youtube_video_url(input_str)
            except Exception as e:
                print(f"Error processing YouTube link: {e}")
                sys.exit(1)
        else:
            raise Exception(f"Invalid YouTube link in the file: {file_path}")
    else:
        raise Exception(f"Unsupported file type: {ext}")

    query = """
        Provide all of the necessary metadata from this pdf containing information regarding
        results from psychology surveys. Put all of the metadata into a csv format suitable for 
        microsoft excel. The format of the csv should look like this: 
        
        questions/statements,survey_name,psychological_facet,facet_num,date,test_format
        question1,name,ex_facet,2,01/04/24,ex_format
        .
        .
        .
        
        Make sure to include the column labels. If the csv already has the columns properly labeled in the first row,
        do not write the column labels again.
    """

    response, docs = get_response_from_query(db, query)

    output_file_path = "../out/output.csv"
    with open(output_file_path, "a", encoding="utf-8") as file:
        file.write(response)
    print(f"Cleaned response has been saved to: {output_file_path}")


def clear_csv_file():
    """
    Clears the content of the specified CSV file.

    Parameters:
    file_path (str): The path to the CSV file to be cleared.
    """
    with open("../out/output.csv", "w", encoding="utf-8") as file:
        pass
    print(f"The file output.csv has been cleared.")


if __name__ == "__main__":
    # clear_csv_file()

    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[-1]
    process_file(file_path)
