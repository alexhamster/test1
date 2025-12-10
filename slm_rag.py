# pip install unsloth
import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import argparse
from unsloth import FastModel
from transformers import TextStreamer

BASEDIR_PATH = os.path.dirname(__file__)
PROMT_PATH = os.path.join(BASEDIR_PATH, "promt.txt")
QUESTIONS_PATH = os.path.join(BASEDIR_PATH, "questions.txt")
CHROMADB_PATH = "./chroma_store"
CHROMA_DIR = os.path.join(BASEDIR_PATH, CHROMADB_PATH)
COLLECTION_NAME = "files_chunks"
#DATASTORE_PATH = "./storage"
DATASTORE_PATH = os.path.join(BASEDIR_PATH, "./storage") # TODO edit

# if N_CHUNKS = -1 will use recomended number of cunks for short context window, else will use your number
N_CHUNKS = -1

EMBEDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# value from 0.0 to 1.0. The higher the value, the more the model hallucinates
MODEL_TEMPERATURE = 1.0


def get_vector_storage():
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )
    return collection


def add_chunks_to_db(collection, documents, metadatas, ids):
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Chunks added: {len(documents)}")
    else:
        print("No files .md found")


def create_chunks(markdown_dir, collection, global_chunk_index = 0):
    
    md_files = glob.glob(os.path.join(markdown_dir, "**", "*.md"), recursive=True)
    
    # Chunk splitter setup. It sepatates files based on articles. 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
        if not text.strip():
            continue
        
        #if len(text) < 50: # skip small files
        #    print(f"Skipped {file_path}")
        #    continue
        
        documents, metadatas, ids = [], [], []
        chunks = splitter.split_text(text)
        
        application_name = file_path.split("/")[-2]
        
        for local_chunk_index, chunk in enumerate(chunks):
            doc_id = f"md_{global_chunk_index}"

            # prefix and postfix to add info about chunk document
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
        
def populate_vector_storage():
    
    collection = get_vector_storage()
    print("DEBUG: db created")
    
    create_chunks(DATASTORE_PATH, collection, 0)
    print("DEBUG: chunks created and added")


def find_chunks(query, collection, n_results):
    
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "ids": result["ids"][0],
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0],
        "distances": result["distances"][0],
    }
    
def prepare_promt(query, n_chunks):
    
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
        
    full_promt = promt.replace("<CONTEXT>", context).replace("<QUERY>", query).replace("\n\n", "\n")
    #print(f"full_promt LEN: {len(full_promt)}")
    #print(full_promt)
    return full_promt

def init_model(model_alias="gemma"):
    
    model_url = ""
    max_seq_len = 2048
    max_chunks_for_short_context_win = 1
    
    if(model_alias == "gemma"):
        model_url = "nhannguyen2730/gemma3-4b-qlora"
        max_seq_len = 8192
        max_chunks_for_short_context_win = 10
    elif(model_alias == "phi"):
        model_url = "nhannguyen2730/phi-3-mini-instruct-qlora-tc"
        max_seq_len = 4096
        max_chunks_for_short_context_win = 5
    else:
        print("Unknown model alias")
        return None, None, 0
        
    model, tokenizer = FastModel.from_pretrained(
    model_name = model_url,
    max_seq_length = max_seq_len,
    load_in_4bit = True,
    )
    
    return model, tokenizer, max_chunks_for_short_context_win

def run_slm_one_query(query, model, tokenizer, n_chunks):

    #model, tokenizer, n_chunks = init_model(model_alias)
    
    if(N_CHUNKS != -1):
        n_chunks = N_CHUNKS
    
    promt = prepare_promt(query, n_chunks)
    
    messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : promt,}]
    }]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 128, # Increase for longer outputs!
        temperature = MODEL_TEMPERATURE, top_p = 0.95, top_k = 64,
    )

    # decoded answer of the model
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return generated_text, query, promt

def run_slm_many_queries(model, tokenizer, n_chunks):
    
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            generated_text, query, _ = run_slm_one_query(line.strip(), model, tokenizer, n_chunks)
            print(f"QUESTION {i}: {query} \n ANSWER {i}: {generated_text}")
       
            
def one_question(query, model_alias):
    model, tokenizer, n_chunks = init_model(model_alias)
    generated_text, _, _ = run_slm_one_query(query, model, tokenizer, n_chunks)
    print(f"QUESTION: {query} \n ANSWER: {generated_text}")   
            
def questions_from_file(model_alias):
    model, tokenizer, n_chunks = init_model(model_alias)
    run_slm_many_queries(model, tokenizer, n_chunks)
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-init",
        action="store_true",
        help="Make chunks from files and init vector storage. Needs to be run once before usage."
    )

    parser.add_argument(
        "-one",
        action="store_true",
        help="To ask slm one question"
    )

    parser.add_argument(
        "-multi",
        action="store_true",
        help="Iteration through the list of questions from questions.txt"
    )

    parser.add_argument(
        "-m",
        type=str,
        help="Name of the SLM. gemma or phi"
    )
    
    parser.add_argument(
        "-q",
        type=str,
        help="Question to the model (required for -one only)"
    )

    args = parser.parse_args()

    if args.one:
        if not args.m or not args.q:
            parser.error("arguments -m (model name: gemma or phi) and -q (question) are required when using -one")

    if args.multi:
        if not args.m:
            parser.error("argument -m (model name: gemma or phi) is required when using -multi")

    if args.one and args.multi:
        parser.error("arguments -one and -multi cannot be used together")

    if args.init:
        populate_vector_storage()
    elif args.one:
        one_question(args.q, args.m)
    elif args.multi:
        questions_from_file(args.m)
    else:
        parser.error("Please select argument")


if __name__ == "__main__":
    main()