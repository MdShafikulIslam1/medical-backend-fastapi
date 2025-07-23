
from fastapi import FastAPI,Request
from langchain_pinecone import PineconeVectorStore
import uvicorn
from src.helper import download_hugging_face_embeddings, language_detection,translate
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*],  # বা ["*"] সব origin এর জন্য
    allow_credentials=True,
    allow_methods=["*"],  # বা ["GET", "POST", "PUT", ...]
    allow_headers=["*"],
)

embeddings = download_hugging_face_embeddings()
index_name = "medical-bot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})
chatModel = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.post("/medi-conversation")
async def medi_conversation(request: Request):
    data = await request.json()
    query = data["query"]
    translated_query = translate(text=query, target_language="English")
    response = rag_chain.invoke({"input": translated_query})
    language = language_detection(query)
    translated_answer = translate(text=response["answer"], target_language=language)
    return {"msg": "success", "data": {"query_language": language, "query_translated": translated_query, "user": query, "bot": translated_answer}}



    
if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=True,log_level="debug")  # no reload here