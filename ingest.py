import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Veri Yükleme
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

# 2. Veri Parçalama ve Kaydetme
def ingest_data():
    print("Veriler yükleniyor...")
    raw_docs = load_documents("./data")
    
    # Metni parçalara böl (Chunking) - Context window taşmaması için kritik
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Embedding ve DB Kaydı
    print(f"{len(chunks)} parça vektör veritabanına işleniyor...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Veritabanını diske kaydet (persistent)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("İşlem tamam! Veritabanı hazır.")

if __name__ == "__main__":
    ingest_data()
