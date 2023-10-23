from dotenv import load_dotenv
# from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import fitz
import os
import json
import shutil
from langchain.llms import OpenAI, GPT4All


# Get the list from the documents directory
def getfiles():
    directory_path = "./documents"

    # List all files in the directory
    file_names = os.listdir(directory_path)

    # Get each of the filenames in the directory and summarise
    for filename in file_names:
        summarise(filename)


# summarise each document
def summarise(filename):
    # API Keys for OPENAI
    LLM_KEY = os.environ.get("OPENAI_API_KEY")

    text = ""

    # Directory path
    directory_path = "./documents"

    # Combine the directory and filename using os.path.join()
    source_file = os.path.join(directory_path, filename)

    #PYPDFLoader loads a list of PDF Document objects
    loader = PyPDFLoader(source_file)

    pages = loader.load()

    for page in pages:
        text+=page.page_content
    text = text.replace('\t', ' ')

    print(len(text))

    #splits a long document into smaller chunks that can fit into the LLM's
    #model's context window
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100
    )
    #create the documents from list of texts
    texts = text_splitter.create_documents([text])

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary with key learnings\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with detailed context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    # Define the LLM
    # here we are using OpenAI's ChatGPT
    # llm=select_llm()
    # llm = OpenAI()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, request_timeout=120)

    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    refine_outputs = refine_chain({'input_documents': texts})
    print(refine_outputs['output_text'])
    savedata(filename, refine_outputs['output_text'])


# Save summary to a file
def savedata(filename, summary):
    data = {
        "summary": summary
    }

    # New filename with .json extension
    file_name = os.path.splitext(filename)[0] + ".json"

    # Open the file in write mode and save the data as JSON
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data has been saved to {file_name}")


def relocatefile(file_name):
    # Directory path
    directory_path = "./documents"

    # Combine the directory and filename using os.path.join()
    source_file = os.path.join(directory_path, file_name)

    # Destination directory (the directory where you want to move the file)
    destination_directory = "./summarised"

    # Combine the destination directory with the source file name to get the new file path
    new_file_path = os.path.join(destination_directory, os.path.basename(source_file))

    # Move the file to the destination directory
    shutil.move(source_file, new_file_path)

    print(f"File has been moved to {new_file_path}")


# Action the file summarisation
getfiles()
