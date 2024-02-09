import pandas as pd
from itertools import chain
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import os

from rag import AIDTRag, STRetriever, AIDTEvaluator

# In this experiment, we run (i.e., replicate) the results of the human developers using
# the original AIDT tool to retrieve technologies. 
# Both the retrieval and ranking functions are exercised. 
# The results are not based on the RAG (i.e., the current Github database) but rather
# on the meta-search strategy (as formulated in the WICSA paper)
# The results, however, are explained by invoking ChatGPT
# ---------------------------------------------------------------------------------------------

# Configuring OpenAI (GPT)
OPENAI_API_KEY = "your OPENAI API KEY goes here"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print()

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
ST_RETRIEVAL_RANKINGS = "./data/bordafuse_rankings.csv"
QUERIES = "./data/queries.csv"
# Outputs
OUTPUT_RANKINGS = "./data/rankings_experiment2b.csv"
OUTPUT_METRICS = "./data/metrics_experiment2b.csv"

TOP_K = 10
selector = 'borda_fuse' # 'humans' # 'npmjs' # 'borda_fuse'

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#df['description'] = df['description'].fillna(value="")
rankings_df = pd.read_csv(ST_RETRIEVAL_RANKINGS)
rankings_df[selector] = rankings_df[selector].apply(eval) # It is a string, so we need to convert it to a list
rankings_df.sort_values('query', inplace=True)

# Counting the unique technologies retrieved from the rankings
elements = [x for x in rankings_df[selector].tolist()] 
unique_elements = set(chain(*elements))
print(len(unique_elements), "unique technologies in the rankings / "+selector)
all_tecnologies = set(technologies_df['name'].tolist())
print(len(unique_elements.intersection(all_tecnologies)), "technologies (from rankings) are in the Github database")

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

# Configuration of a fake retriever with the human rankings (based on the AIDT tool)
st_retriever = STRetriever.from_documents(documents, rankings_df, col=selector)

rag = AIDTRag(technologies_df, k=TOP_K, retriever=st_retriever)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
rag.set_llm(llm)

# These are the reference queries for the experiment
userstories_df = pd.read_csv(QUERIES)
#userstories_df = userstories_df.head(5)
print() 

# Run the RAG for all the queries
list_dfs = []
ignoreLLM = False
with get_openai_callback() as cb:
    for index, row in userstories_df.iterrows():
        print(index, "-"*100)
        query = row['query']
        print("Query:", query)
        # Both retrieval and re-ranking are needed in this experiment
        reranking_json = rag.execute(query, rerank='gbrank', explain=True)
        print("Retrieval & re-ranking + explanation:", TOP_K)
        ranking_df = pd.read_json(StringIO(reranking_json))
        if ranking_df.shape[1] == 1: # It's a fake LLM (no responses)
            print(ranking_df)
            ignoreLLM = True
        elif ranking_df.shape[0] > 0:
            print(ranking_df)
            print(ranking_df[['package_name', 'adjectives', 'description']])
            ranking_df['query'] = str(query)
            ranking_df['who'] = selector
            list_dfs.append(ranking_df)
        else:
            print("Ranking seems empty")
            print(ranking_df)
        print()
    print("-"*100)
print(cb)
print()

if not ignoreLLM:
    results_df = pd.concat(list_dfs, ignore_index=True)
    results_df.to_csv(OUTPUT_RANKINGS, index=False) # Save the rankings for further analysis
else:
    output_rankings_df = pd.DataFrame(columns=['query', selector])
    print("Reading the rankings from previously saved results:", OUTPUT_RANKINGS)
    results_df = pd.read_csv(OUTPUT_RANKINGS)
    results_df['package_name'] = results_df['package_name'].str.lower()

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
