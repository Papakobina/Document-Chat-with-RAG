import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == '__main__':
    load_dotenv()
    print("Ingesting data...")
    
    loader = TextLoader("./mediumblog1.txt", encoding='utf-8')
    document = loader.load()
    
    print("Splitting text...")
    
    text_Splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_Splitter.split_documents(document)
    print(f"created {len(texts)} text chunks")
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Ingesting data...")
    
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    
    print("Ingestion complete")