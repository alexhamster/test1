import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from ollama import Client
import argparse

BASEDIR_PATH = os.path.dirname(__file__)
PROMT_PATH = os.path.join(BASEDIR_PATH, "promt.txt")
QUESTIONS_PATH = os.path.join(BASEDIR_PATH, "questions.txt")
CHROMADB_PATH = "./chroma_store"
CHROMA_DIR = os.path.join(BASEDIR_PATH, CHROMADB_PATH)
COLLECTION_NAME = "files_chunks"
#DATASTORE_PATH = "./storage"
DATASTORE_PATH = os.path.join(BASEDIR_PATH, "./test_storage") # TODO edit
N_CHUNKS = 10



def get_vector_storage():
    '''
    init choma db with embeding model.
    
    :param chroma_dir_path: path to chromaDB database
    :param collection_name: name of collection to store chunks
    :return: collection object to work with DB
    
    '''
    # embeding model to transform chunks into vectors
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )
    
    return collection

def create_chunks(markdown_dir, collection, global_chunk_index = 0):
    
    '''
    creating chunks from .md files. This function iterates folders starting from <markdown_dir>
    and creates chunks from each file. 
    
    :param markdown_dir: path do markdown files to 
    :param global_chunk_index: starting global index to enumerate chunks
    :return:    tuple of:
                documents - parts of text
                metadatas - data about file path, file name and local chunk index
                ids - global chunk index
    
    '''
    
    md_files = glob.glob(os.path.join(markdown_dir, "**", "*.md"), recursive=True)
    

    # Chunk splitter setup. It sepatates files based on articles. 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""],
    )
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
        if not text.strip():
            continue
        
        if len(text) < 50: # skip small files
            print(f"Skipped {file_path}")
            continue
        
        documents, metadatas, ids = [], [], []
        chunks = splitter.split_text(text)
        
        application_name = file_path.split("/")[-2]
        
        for local_chunk_index, chunk in enumerate(chunks):
            doc_id = f"md_{global_chunk_index}"

            chunk_with_prefix = f"BEGIN {application_name} >>> \n\n{chunk} \n\n <<< {application_name} END"
            
            documents.append(chunk_with_prefix)
            metadatas.append({
                "path": file_path,
                "filename": os.path.basename(file_path),
                "chunk": local_chunk_index,   # chunk number inside file
                "application": application_name,

            })
            
            ids.append(doc_id)
            global_chunk_index += 1
            
        add_chunks_to_db(collection, documents, metadatas, ids)
            

    return documents, metadatas, ids

def add_chunks_to_db(collection, documents, metadatas, ids):
    
    '''
    add chunks to current ChromaDB
    
    :param collection: ChromaDB object
    :param documents, metadatas, ids - chunks information
    
    '''
    
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Chunks added: {len(documents)}")
    else:
        print("No files .md found")
      
             
def find_chunks(query, collection, n_results: int = 1):
    """
    Search closest document in the ChromaDB based on text query

    :param query: String-request
    :param collection: ChromaDB collection object
    :param n_results: Number of the best chunks to return 
    :return: Dict with chunks, metadata and distance metric (the smaller the distance, the better the chunk fits the query)
        dict format:     
        # {
        #   "ids": [[...]],
        #   "documents": [[...]],
        #   "metadatas": [[...]],
        #   "distances": [[...]],
        # }
    """
    
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    #for i in result["metadatas"][0]:
    #    print(i["application"])
    

    return {
        "ids": result["ids"][0],
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0],
        "distances": result["distances"][0],
    }


def prepare_promt(query, n_chunks=10):
    '''
    Prepare full promt
    '''
    
    # get ChromdDB object
    collection = get_vector_storage()
    
    promt = ""
    with open(PROMT_PATH, "r", encoding="utf-8") as f:
        promt = f.read()
    
    # Searching for context based on query
    result = find_chunks(query, collection, n_chunks)
    
    context = ""
    for i, doc_id in enumerate(result["ids"]):
        context += "\n\n" + result["documents"][i]
        
    #print("================")    
    #print(context.replace("\n\n", "\n"))
    #print(f"CONTEXT LEN: {len(context)}")
    #print("================")    
    
    full_promt = promt.replace("<CONTEXT>", context).replace("<QUERY>", query)
    
    return full_promt


def query_ollama_with_context(query, model_name="gpt-oss:20b"):
    """
    
    Makes request to local slm model adding context.
    
    :param query: Raw user request
    :param model_name: slm model name which is launched in Ollama
    :return: String answer from the model
    """
    # Default Ollama url
    client = Client(host="http://localhost:11434")
    
    promt = prepare_promt(query, N_CHUNKS)
    
    response = client.generate(
        model=model_name,
        prompt=promt
    )

    return response["response"]


def populate_vector_storage():
    '''
    Setup ChromaDB, make chunks from .md files. 
    '''
    collection = get_vector_storage()
    print("db created")
    
    documents, metadatas, ids = create_chunks(DATASTORE_PATH, collection, 0)
    print("chunks created")

    #add_chunks_to_db(collection, documents, metadatas, ids)
    print("chunks added to vector storage")
    
       
def answer_questions_from_file():
    
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            #print(i, line.strip())
            answer = query_ollama_with_context(line.strip())
            print(answer)
            
    
def answer_one_question(query):
    
    answer = query_ollama_with_context(query)
    print(answer) 
    return answer   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", action="store_true", help="Make chunks from files and init vector storage. Needs to be run once before usage.")
    parser.add_argument(
        "-one",
        metavar="QUERY",
        type=str,
        required=False,
        help="Ask slm one question"
    )
    parser.add_argument("-multi", action="store_true", help="Iteration through the list of questions from questions.txt")
    args = parser.parse_args()

    if args.init:
        populate_vector_storage()
    elif args.one:
        answer_one_question(args.one)
    elif args.multi:
        answer_questions_from_file()
    else:
        parser.error("Please select argument")


if __name__ == "__main__":
    main()

