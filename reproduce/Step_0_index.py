# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio  
from minirag import MiniRAG  
from minirag.kg.postgres_impl import PostgreSQLDB
from minirag.llm.openai import (
    openai_complete,
)
from minirag.llm.openai import openai_embed  
from minirag.utils import EmbeddingFunc


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen3-instruct")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()


if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen3-instruct":
    LLM_MODEL = "qwen_qwen3-4b-instruct-2507"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)

os.environ["AGE_GRAPH_NAME"] = "minirag_graph" 

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Step 1: Create PostgreSQL DB instance  
db = PostgreSQLDB(  
    config={  
        "host": "192.168.1.111",
        "port": 5455,           # Matches your docker -p 5455:5432
        "user": "postgres",     # Matches POSTGRES_USER
        "password": "postgres", # Matches POSTGRES_PASSWORD
        "database": "minirag",  # Matches POSTGRES_DB
        "workspace": "default"   
    }  
)  
  
# Step 2: Initialize the database connection  
# Step 2: Initialize the database connection  
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(db.initdb())
loop.run_until_complete(db.check_tables())  

# Step 3: Create MiniRAG with PostgreSQL storage
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,

    # --- POSTGRES CONFIGURATION START ---
    # Tell MiniRAG to use the Postgres implementations
    kv_storage="PGKVStorage", 
    vector_storage="PGVectorStorage", 
    graph_storage="PGGraphStorage", 
    doc_status_storage="PGDocStatusStorage",
    
    # Connection credentials matching your Docker command
    #addon_params={
    #    "host": "192.168.1.111",
    #    "port": 5455,           # Matches your docker -p 5455:5432
    #    "user": "postgres",     # Matches POSTGRES_USER
    #    "password": "postgres", # Matches POSTGRES_PASSWORD
    #    "database": "minirag",  # Matches POSTGRES_DB
    #    "workspace": "default"  
    #},
    # --- POSTGRES CONFIGURATION END ---
    
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=1000,
        func=lambda texts: openai_embed(  
            texts,  
            model="qwen-embedding-0.6B",
            base_url="http://192.168.1.111:8886/v1",  
            api_key="sk-1"  
        )
        #func=lambda texts: hf_embed(
        #    texts,
        #    tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
        #    embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        #),
    ),
)

# Step 4: Set the database client  
rag.set_storage_client(db)

# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())
