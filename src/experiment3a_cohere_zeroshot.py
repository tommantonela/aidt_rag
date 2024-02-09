import pandas as pd
from io import StringIO
from langchain_community.chat_models import ChatCohere
import os
import pickle
import time

from rag import AIDTRag, AIDTEvaluator

# In this experiment, we use Cohere as a search engine (in the wild) that tries to retrieve
# the different packages for a given need (i.e., query)
# ---------------------------------------------------------------------------------------------

COHERE_API_KEY = "your COHERE API KEY goes here"
print()

INVOCATIONS_PER_MINUTE = 10
SLEEP_TIME = 60 # Delay for 1 minute (60 seconds).

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
QUERIES = "./inputs/queries.csv"
# Outputs
OUTPUT_RANKINGS = "./data/rankings_experiment3a.csv"
OUTPUT_METRICS = "./data/metrics_experiment3a.csv"

TOP_K = 10
selector = 'borda_fuse' # 'humans' # 'npmjs' # 'borda_fuse'

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#technologies_df['description'] = technologies_df['description'].fillna(value="")

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

rag = AIDTRag(technologies_df, k=TOP_K)
llm = ChatCohere(model="command", temperature=0.0) #, max_tokens=256)
rag.set_llm(llm)
selector = 'cohere'

# These are the reference queries for the experiment
userstories_df = pd.read_csv(QUERIES)
#userstories_df = userstories_df.head(3)
print() 
# Run the RAG for all the queries
list_dfs = []
ignoreLLM = False
colsfilter = ['package_name', 'adjectives', 'description']
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
    if (index+1) % INVOCATIONS_PER_MINUTE == 0:
        print("Waiting 1 minute (Cohere API restriction...)")
        time.sleep(60) # Delay for 1 minute (60 seconds)
    
    print(index, "-"*100)
    query = row['query']
    if query not in temporal_dict:
        print("Query:", query)
        # Neither retrieval nor re-ranking are needed in this experiment, only search
        # Prompting style is assumed to be zero-shot
        ranking_json = rag.search(query)
        print("Search (zero-shot):", TOP_K)
        ranking_df = pd.read_json(StringIO(ranking_json))
        if ranking_df.shape[1] == 1: # It's a fake LLM (no responses)
            print(ranking_df)
            ignoreLLM = True
        elif ranking_df.shape[0] > 0:
            ranking_df.rename(columns={"justification": "adjectives", "justifiers": "adjectives", "justify_adjectives": "adjectives", 
                                       "justify": "adjectives", "justify_choices": "adjectives", "package": "package_name"}, inplace=True)
            ranking_df.rename(columns={"year-of-release": "year_of_release", "year": "year_of_release", "release_year": "year_of_release",
                                       "_package_description": "description", "three_adjectives": "adjectives", "qualifiers": "adjectives",
                                        "year_released": "year_of_release", "year-of-release": "release_year"}, inplace=True)
            print(ranking_df[colsfilter])
            ranking_df['query'] = str(query)
            ranking_df['who'] = selector
        else:
            print("Ranking seems empty")
            print(ranking_df)
        temporal_dict[query] = ranking_df
        print()
        if temporal_file is not None:
            with open(temporal_file, 'wb') as output:
                pickle.dump(temporal_dict, output)

print("-"*100)
list_dfs = [df for df in temporal_dict.values() if (df.shape[0] > 0) and (df.shape[1] > 1)]
print(len(list_dfs), "queries processed")
print()

if not ignoreLLM:
    results_df = pd.concat(list_dfs, ignore_index=True)
    results_df['package_name'] = results_df['package_name'].str.lower()
    results_df.to_csv(OUTPUT_RANKINGS, index=False) # Save the rankings for further analysis
else:
    output_rankings_df = pd.DataFrame(columns=['query', selector])
    print("Reading the rankings from previously saved results:", OUTPUT_RANKINGS)
    results_df = pd.read_csv(OUTPUT_RANKINGS)
    results_df['package_name'] = results_df['package_name'].str.lower()
    retrieved_technologies = set(results_df['package_name'].tolist())
    print(len(retrieved_technologies), "technologies retrieved from the rankings (Cohere)")

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
metrics_df = AIDTEvaluator.get_metrics_as_dataframe(query_metrics_dict, who=selector)
print(metrics_df) # This dataframe is useful for generating the boxplots
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print()
