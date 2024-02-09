import pandas as pd
import json
from io import StringIO
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatCohere
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import pickle
import time

from rag import AIDTRag, STRetriever, AIDTEvaluator

# In this experiment, we use GPT as a search engine (in the wild) that tries to retrieve
# the different packages for a given need (i.e., query)
# Note that the need is captures a a couple of kewords, rather than as a user story
# ---------------------------------------------------------------------------------------------

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
QUERIES = "./inputs/queries.csv"
# Outputs
#OUTPUT_RANKINGS = "./data/rankings_experiment3b.csv"
#OUTPUT_METRICS = "./data/metrics_experiment3b.csv"
JSON_OUTPUT_FILE = "./data/json_experiment3b.json"

TOP_K = 3

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#technologies_df['description'] = technologies_df['description'].fillna(value="")

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

my_model = "llama2:7b-chat" #"mistral:7b-instruct"
rag = AIDTRag(technologies_df, k=TOP_K)
llm = ChatOllama(
    model=my_model, temperature=0.0, format="json",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
rag.set_llm(llm)
#ranker = CohereRerank()
#rag.set_reranker(ranker)

selector = 'ollama-'+my_model
print(selector)
# These are the reference queries for the experiment
userstories_df = pd.read_csv(QUERIES)
userstories_df = userstories_df.head(2)
print() 
# Run the RAG for all the queries
python_objects = []
temporal_file = "./temporal_prompts.pkl"
if temporal_file is not None:
    if os.path.exists(temporal_file):
        with open(temporal_file, 'rb') as input:
            temporal_dict = pickle.load(input)
    else:
        temporal_dict = dict()
else:
    temporal_dict = dict()
print(len(temporal_dict.keys()), "queries recovered")

for index, row in userstories_df.iterrows():
    print(index, "-"*100)
    #query = row['user_story']
    query = row['query']
    if query not in temporal_dict:
        print("Query:", query)
        # Both retrieval and re-ranking are needed in this experiment
        ranking_json = rag.search(query)
        print("Search (zero-shot):", TOP_K)
        temporal_dict[query] = ranking_json
        print()
        if temporal_file is not None:
            with open(temporal_file, 'wb') as output:
                pickle.dump(temporal_dict, output)

print("-"*100)

python_objects = [obj_json for obj_json in temporal_dict.values()]
print(len(python_objects), "queries processed")
print()
with open(JSON_OUTPUT_FILE, "w") as f:
     f.write("["+",\n".join(python_objects)+"]")
print()