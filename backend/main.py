import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_engine import process_pdf, answer_query

app = FastAPI(title="RAG Engine Core Backend Service")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives file upload, writes to disk, and runs the RAG ingestion pipeline."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF file structures are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_created = process_pdf(file_path)
        return {"status": "Success", "message": f"Ingested {file.filename} into {chunks_created} shards."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(payload: QueryRequest):
    """Receives queries and routes them directly to the active vector space chain."""
    try:
        answer = answer_query(payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
