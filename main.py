from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import json,os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Initialize key variables and functions
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
FAISS_INDEX_NAME="vdb"
PDF_PATH = "10050-medicare-and-you_0.pdf"

def determine_k_from_query(query: str) -> int:

    """
    Determines the number of chunks (k) to retrieve dynamically.
    Args:
        string : Text which is user query.
    Returns:
        int :  Number of chunks to be retrieved from the vector database.
    """
    query_lower = query.lower()
    if "detail" in query_lower or "explain" in query_lower or "comprehensive" in query_lower:
        return 6
    elif "brief" in query_lower or "summar" in query_lower or len(query) < 20:
        return 1
    else:
        return 4

    
def extract_json_from_string(text_with_json: str) -> dict | None:
    """
    Identifies the first '{' and last '}' in a string and attempts to
    parse the substring between them as a JSON object.

    Args:
        text_with_json: The string that potentially contains a JSON object.

    Returns:
        A Python dictionary if valid JSON is found and parsed, otherwise None.
    """
    first_brace_index = text_with_json.find('{')
    last_brace_index = text_with_json.rfind('}')

    # Check if both braces were found and in the correct order
    if first_brace_index == -1 or last_brace_index == -1 or first_brace_index >= last_brace_index:
        print("Warning: No valid JSON object delimiters '{}' found in the string.")
        return None

    potential_json_string = text_with_json[first_brace_index : last_brace_index + 1]

    # Attempt to parse the sliced string as JSON
    try:
        json_object = json.loads(potential_json_string)
        return json_object
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to decode JSON from extracted substring: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# 2. Loading PDF and creating chunks
try:
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load_and_split()
    print(f"Loaded {len(pages)} pages from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    chunks = text_splitter.split_documents(pages)
    
    print(f"Split document into {len(chunks)} chunks.")
except Exception as e:
    print(f"Error loading PDF: {e}")


# 3. Generate embedding model
print(f"Generating embeddings using GoogleGenerativeAIEmbeddings...")
try:
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("Embedding model initialized.")
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    

# 4. Create the FAISS Vector Store (save model - optionally)
print(f"Creating FAISS vector store...")
try:
    # Create a FAISS vector store from the chunks and embeddings
    db = FAISS.from_documents(chunks, embeddings)
    
    # Save the FAISS index locally
    # db.save_local(FAISS_INDEX_NAME)
    # print(f"FAISS index saved locally as '{FAISS_INDEX_NAME}'.")
    print(f"--- Created FAISS vector  database ---")
    
except Exception as e:
    print(f"Error creating or saving FAISS index: {e}")


# 5. Create LLM model object
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY,temperature =0,
    max_tokens=None,
    timeout=None,
    max_retries=2)


# 6. Create a chain to proess user query
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a polite, helpful assistant who works for a health insurance company and uses companies handbook to answer users queries related to health insurance. Use the following context to answer the question. Also give a confidence score based on how confident you are of the response generated\n\n i want to process a json object from th response so only give response in strict JSON format {{'answer':'..','confidence':score}}"),
    ("human", "Context: {context}\n\nQuestion: {question}\n\n ")
])

output_parser = StrOutputParser()

chain = (prompt | model | output_parser)


# 7. Create FastAPI endpoint 
class UserPrompt(BaseModel):
    query: str

app=FastAPI(title="RAG")

@app.post("/query/")
async def func(data : UserPrompt) -> dict | None:
    '''
    takes a user query and respondes with a JSON containing LLM response, confidence score, source page and chunk length
    Args:
         UserPrompt : User question.

    Returns:
        response : A response object.
    '''
    if data:
        user_question = data.query
    else:
        raise HTTPException(status_code=400, detail="Bad Request.")

    if user_question=="":
        raise HTTPException(status_code=400, detail="Query cannot be empty. Please provide a valid question.")

    retriever = db.as_retriever(search_kwargs={"k": determine_k_from_query(user_question),"score_threshold": 0.75})
    docs = retriever.invoke(user_question)
    context_string=[]
    pages =[]
    for doc in docs:
        page_num = doc.metadata.get('page_label', 'N/A')
        pages.append(str(page_num))
        context_string.append(doc.page_content)
    
    retrieved_context = '\n\n'.join(context_string)
    pages=list(set(pages)) # to remove any page source that comes more than once (because chunks can be part of same page source)
    answer = chain.invoke({"context": retrieved_context, "question": user_question})
    page = ','.join(pages)
    json_obj = extract_json_from_string(answer)
    if json_obj:
        try:
            answer=json_obj['answer']
            score = json_obj['confidence']
        except:
            score = 'unable to fetch'
    else:
        score= "unable to fetch"
        
    response = {"answer":answer,
                "source_page": page,
                "confidence_score": score,
                "chunk_size": len(retrieved_context)
                }

    return response

