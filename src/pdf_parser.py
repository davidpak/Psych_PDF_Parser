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

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

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

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

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


file_path = "../test_files/linear_regression.pdf"
query = "Take notes on this video in this format. Make sure that you properly newline throughout the page: """"
    # [Title]

    ## General Overview 
    [Provide a brief summary or introduction of the topic.]

    ## Key Concepts

    - **Concept 1:**
        - [Brief description or explanation of the first key concept.]
    - **Concept 2:**
        - [Brief description or explanation of the second key concept.]
    - **Concept 3:**
        - [Brief description or explanation of the third key concept.]

    ## Section by Section Breakdown

    ### 1. Section One Title

    - [Detailed content or information related to the first section.]

    ### 2. Section Two Title

    - [Detailed content or information related to the second section.]

    ### 3. Section Three Title

    - [Detailed content or information related to the third section.]

    ### n. Section n Title

    - [Detailed content or information related to the nth section]

    ## Additional Information

    - [Include any additional points, tips, or related information.]

    ## Helpful Vocabulary

    - **Term 1:**
        - [Definition or explanation of the first term.]
    - **Term 2:**
        - [Definition or explanation of the second term.]
    - **Term 3:**
        - [Definition or explanation of the third term.]
    - **Term n:**
        - [Definition or explanation of the nth term.]

    ## Explain it to a 5th grader:

    [Provide an explanation about this topic suitable for a 5th grader]

    ## Conclusion

    [Summarize the key takeaways or concluding remarks.]
    """

db, docs = create_db_from_pdf(file_path)
response, docs = get_response_from_query(db, query)
print(response)

