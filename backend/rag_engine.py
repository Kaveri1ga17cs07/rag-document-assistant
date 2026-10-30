import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma  # Updated from langchain_community.vectorstores
from langchain_classic.chains import create_retrieval_chain  # Moved to langchain_classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # Moved to langchain_classic
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# Global variables for persistence inside runtime memory
DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
vector_store = None

def process_pdf(file_path: str):
    """Parses PDF, chunks text, and builds the Chroma vector database index."""
    global vector_store
    
    # 1. Load the text from the PDF file
    loader = PyPDFLoader(r"C:\Users\shrid\Downloads\rag_evaluation.pdf")
    docs = loader.load()
    
    # 2. Split text into manageable token chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Embed text fragments into local Chroma Vector DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return len(splits)

def answer_query(question: str):
    """Retrieves context from vector store and uses LLM to answer user question."""
    global vector_store
    
    # Load existing persistent database index if none exists in dynamic memory
    if vector_store is None:
        embeddings = OpenAIEmbeddings()
        if os.path.exists(DB_DIR):
            vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        else:
            return "No documents uploaded yet. Please upload a PDF before asking questions."
            
    # 4. Initialize RAG components (Retriever & Generation Model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    
    # Define prompt template enforcing context boundaries
    system_prompt = (
        "You are an expert AI assistant answering questions strictly based on the provided context.\n"
        "If you do not know the answer based on the context, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Build and execute the RAG operational chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": question})
    return response["answer"]
