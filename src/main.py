import sys
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
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


def create_db_from_csv_file(csv_file):
    loader = CSVLoader(csv_file)
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
    template = f"""
        You are a helpful assistant that that provides metadata on pdfs regarding psychology research surveys
         using this transcript: {docs_page_content}.

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
    csv_path = "../out/output.csv"
    # db_csv, docs_csv = create_db_from_csv_file(csv_path)
    # csv_content = " ".join([d.page_content for d in docs_csv])

    if ext == ".pdf":
        db, docs = create_db_from_pdf(file_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")

    query = f"""
        Provide all of the necessary metadata from this pdf containing information regarding
        results from psychology surveys. Put all of the metadata into a csv format suitable for 
        microsoft excel. The csv currently looks like this: 
        
        questions/statements,survey_name,psychological_facet,facet_num,date,test_format
        .
        .
        .
        
        Do not write the column labels, just write the corresponding metadata from the PDF into its respective column. End
        the output with a newline.
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
