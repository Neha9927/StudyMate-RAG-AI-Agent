#1. load raw pdf
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()


data_Path="Data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                   glob='*pdf',
                   loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
documents=load_pdf_files(data=data_Path)
print ("length of pdf docs pages:",len(documents))

#2. create chunk
def create_chunk(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunk=text_splitter.split_documents(extracted_data)
    return text_chunk
text_chunk=create_chunk(extracted_data=documents)
print ("length of text chunks:",len(text_chunk))

#3. create vector embeding
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_embedding_model()
#4. store embedding in FAISS
db_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(text_chunk,embedding_model)
db.save_local(db_FAISS_PATH)


