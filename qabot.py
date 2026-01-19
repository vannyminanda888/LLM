from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import gradio as gr

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM
def get_llm():
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.2,
    }
    project_id = "skills-network"

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm


## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(data)

    chunks = [
        doc for doc in chunks
        if doc.page_content and doc.page_content.strip()
    ]

    return chunks


## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512
    }

    return WatsonxEmbeddings(
        model_id="ibm/slate-embedding-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )


## Vector DB
def vector_database(chunks):
    if not chunks:
        raise ValueError("No valid text chunks found in document.")

    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    print(f"Vector DB created: {vectordb}")
    return vectordb


## Retriever
def retriever(file):
    splits = document_loader(file)
    print(f"Documents loaded: {len(splits)}")
    
    chunks = text_splitter(splits)
    print(f"Chunks created: {len(chunks)}")
    
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    # response = qa.invoke({"query": query})
    return qa.run(query)


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here...",
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering (RAG with watsonx)",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
