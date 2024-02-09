import pandas as pd
from itertools import chain
import json
from io import StringIO
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys

from rag import AIDTRag, STRetriever, AIDTEvaluator

# In this experiment, we use GPT as a search engine (in the wild) that tries to retrieve
# the different packages for a given need (i.e., query)
# Note that the need is captures a a couple of kewords, rather than as a user story
# ---------------------------------------------------------------------------------------------

# Configuring OpenAI (GPT)
ENV_PATH = sys.path[0]+'/andres.env'
print("Reading OPENAI config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print()

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
QUERIES = "./inputs/queries.csv"
# Outputs
OUTPUT_RANKINGS = "./data/rankings_experiment3.csv"
OUTPUT_METRICS = "./data/metrics_experiment3.csv"

selector = 'gpt_zeroshot'
TOP_K = 10

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#technologies_df['description'] = technologies_df['description'].fillna(value="")

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

rag = AIDTRag(technologies_df, k=TOP_K)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
rag.set_llm(llm)

# These are the reference queries for the experiment
userstories_df = pd.read_csv(QUERIES)
userstories_df = userstories_df.head(0)

print() 
# Run the RAG for all the queries
list_dfs = []
ignoreLLM = False
with get_openai_callback() as cb:
    for index, row in userstories_df.iterrows():
        print(index, "-"*100)
        #query = row['user_story'] # If this field is used as they query, results vary considerably
        query = row['query']
        print("Query:", query)
        # Neither retrieval nor re-ranking are needed in this experiment, only search
        # Prompting style is assumed to be zero-shot
        ranking_json = rag.search(query)
        print("Search (zero-shot):", TOP_K)
        #print(ranking_json)
        ranking_df = pd.read_json(StringIO(ranking_json))
        if ranking_df.shape[1] == 1: # It's a fake LLM (no responses)
            print(ranking_df)
            ignoreLLM = True
        elif ranking_df.shape[0] > 0:
            #print(ranking_df)
            print(ranking_df[['package_name', 'adjectives', 'description']])
            ranking_df['query'] = str(query)
            ranking_df['who'] = selector
            list_dfs.append(ranking_df)
        else:
            print(ranking_df)
        print()
    print("-"*100)
print(cb)
print()
if not ignoreLLM:
    results_df = pd.concat(list_dfs, ignore_index=True)
    results_df['package_name'] = results_df['package_name'].str.lower()
    results_df.to_csv(OUTPUT_RANKINGS, index=False) # Save the rankings for further analysis
else:
    output_rankings_df = pd.DataFrame(columns=['query', selector])
    # TODO: Re-ingest the CSV (without running the LLM) in order to compute the metrics
    print("Reading the rankings from previously saved results:", OUTPUT_RANKINGS)
    results_df = pd.read_csv(OUTPUT_RANKINGS)
    results_df['package_name'] = results_df['package_name'].str.lower()
    retrieved_technologies = set(results_df['package_name'].tolist())
    print(len(retrieved_technologies), "technologies retrieved from the rankings (GPT)")

output_rankings_df = results_df.groupby('query')['package_name'].apply(list).reset_index()
output_rankings_df.columns = ['query', selector]
output_rankings_df.sort_values('query', inplace=True)
print(output_rankings_df)

print("Running evaluation metrics...", selector, "versus Ground Truth")
gt_df = pd.read_csv(GROUND_TRUTH)
gt_df['hits'] = gt_df['hits'].apply(eval)
evaluator = AIDTEvaluator(gt_df)
query_metrics_dict = evaluator.get_metrics(output_rankings_df)
metrics_dict = AIDTEvaluator.get_metrics_by_type(query_metrics_dict)
# print(json.dumps(metrics_dict, indent=4))
metrics_df = AIDTEvaluator.get_metrics_as_dataframe(query_metrics_dict, who=selector)
print(metrics_df) # This dataframe is useful for generating the boxplots
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print()
