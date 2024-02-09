import pandas as pd
from io import StringIO
from langchain_community.chat_models import ChatCohere
import os
import pickle
import time

from rag import AIDTRag, STRetriever, AIDTEvaluator

# In this experiment, we use GPT as a search engine (in the wild) that tries to retrieve
# the different packages for a given need (i.e., query)
# ---------------------------------------------------------------------------------------------

# Configuration of Cohere LLM
COHERE_API_KEY = "your COHERE API KEY goes here"
print()

INVOCATIONS_PER_MINUTE = 5
SLEEP_TIME = 60 # Delay for 1 minute (60 seconds).

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
QUERIES = "./inputs/queries.csv"
ST_RETRIEVAL_RANKINGS = "./data/bordafuse_rankings.csv"
# Outputs
OUTPUT_RANKINGS = "./data/rankings_experiment4a.csv"
OUTPUT_METRICS = "./data/metrics_experiment4a.csv"

TOP_K = 10
selector = 'borda_fuse' # 'humans' # 'npmjs' # 'borda_fuse'

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#df['description'] = df['description'].fillna(value="")
rankings_df = pd.read_csv(ST_RETRIEVAL_RANKINGS)
rankings_df[selector] = rankings_df[selector].apply(eval) # It is a string, so we need to convert it to a list
rankings_df.sort_values('query', inplace=True)

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

# Configuration of a fake retriever with the human rankings (based on the AIDT tool)
st_retriever = STRetriever.from_documents(documents, rankings_df, col=selector)

rag = AIDTRag(technologies_df, k=TOP_K, retriever=st_retriever)
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
    print(index, "-"*100)
    query = row['query']
    if query not in temporal_dict:
        if (index+1) % INVOCATIONS_PER_MINUTE == 0:
            print("Waiting 1 minute (Cohere API restriction...)")
            time.sleep(60) # Delay for 1 minute (60 seconds)
        print("Query:", query)
        # Only retrieval is needed in this experiment
        reranking_json = rag.execute(query, rerank='cohere', explain=True)
        # if not reranking_json.startswith("["):
        #     reranking_json = "["+reranking_json+"]"
        print("Retrieval + cohere:", TOP_K)

        ranking_df = pd.read_json(StringIO(reranking_json))
        if ranking_df.shape[1] == 1: # It's a fake LLM (no responses)
            print(ranking_df)
            ignoreLLM = True
        elif ranking_df.shape[0] > 0:
            ranking_df.rename(columns={"justification": "adjectives", "justifiers": "adjectives", "justify_adjectives": "adjectives", "qualities": "adjectives",
                                       "justify": "adjectives", "justify_choices": "adjectives", "justifications": "adjectives", "justification_adjectives": "adjectives",
                                       "package-name": "package_name", "package": "package_name", "qualifiers": "adjectives" }, inplace=True)
            ranking_df.rename(columns={"year-of-release": "year_of_release", "year": "year_of_release", "year-released": "year_of_release",
                                        "year_released": "year_of_release", "year-of-release": "release_year"}, inplace=True)
            print(ranking_df[colsfilter])
            ranking_df['query'] = row['query'] #str(query)
            ranking_df['who'] = selector
            print("Grounding check:", rag.check_grounding(query, ranking_df['package_name'].tolist()))
        else:
            print("Ranking seems empty")
            print(ranking_df)
        ranking_df.reset_index(inplace=True, drop=True)
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
