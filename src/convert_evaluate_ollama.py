import pandas as pd
import json
import rag

GROUND_TRUTH = "./data/queries_x_hits.csv"
QUERIES = "./inputs/queries.csv"

# JSON_RANKINGS = "./outputs/json_experiment3b_ollama.json"
# OUTPUT_RANKINGS = "./data/rankings_experiment3b.csv"
# OUTPUT_METRICS = "./data/metrics_experiment3b.csv"
JSON_RANKINGS = "./outputs/json_experiment4b_ollama.json"
OUTPUT_RANKINGS = "./data/rankings_experiment4b.csv"
OUTPUT_METRICS = "./data/metrics_experiment4b.csv"


def convert_json_to_dataframe(filename, queries, output=None):
    with open(filename) as f:
        d = json.load(f)
        #print(d)
    userstories_df = pd.read_csv(queries)

    #print()
    json_str = json.dumps(d, indent=4)
    #print(json_str)
    print("list size:", len(d))
    list_dfs = []
    #pd.read_json('./outputs/json_experiment3b_ollama.json')
    for idx, r in enumerate(d):
        ignore_row = False
        #print(idx, r)
        if 'packages' in r:   
            ranking_dict = r['packages']
        elif 'suggestions' in r: 
            ranking_dict = r['suggestions']
        elif 'suggestedPackages' in r:
            ranking_dict = r['suggestedPackages']
        elif 'questions' in r:
            ranking_dict = r['questions']
        else:
            print("Ignoring row", idx, r)
            ignore_row = True #ranking_dict = r['scraper']
    
        if not ignore_row:
            ranking_df = pd.DataFrame(ranking_dict) # package_name,description,url,year_of_release,adjectives,pros,cons,query,who
            if "rating" in ranking_df.columns:
                ranking_df.drop("rating", axis=1, inplace=True)   
            ranking_df.rename(columns={"year": "year_of_release", "name": "package_name", "justify": "adjectives", 
                      "packageName": "package_name", "shortDescription": "description", "yearOfRelease": "year_of_release", 
                      "justifyChoice": "adjectives", "justify_choice": "adjectives"}, inplace=True)
            #print(userstories_df.loc[idx,'query'])
            ranking_df['query'] = userstories_df.loc[idx,'query']
            #print(ranking_df.shape, ranking_df.columns)
            #print(ranking_df)
            list_dfs.append(ranking_df)
    
    results_df = pd.concat(list_dfs)
    results_df['package_name'] = results_df['package_name'].str.lower()
    if output is not None:
        results_df.to_csv(OUTPUT_RANKINGS, index=False) # Save the rankings for further analysis
    
    return results_df

print()
print("Reading rankings (JSON) from ... ", JSON_RANKINGS)
results_df = convert_json_to_dataframe(JSON_RANKINGS, QUERIES, OUTPUT_RANKINGS)
selector = 'ollama-llama2:7b-chat'
output_rankings_df = results_df.groupby('query')['package_name'].apply(list).reset_index()
output_rankings_df.columns = ['query', selector]
output_rankings_df.sort_values('query', inplace=True)
print(output_rankings_df.head(10))

print()
print("Running evaluation metrics...", selector, "versus Ground Truth")
gt_df = pd.read_csv(GROUND_TRUTH)
gt_df['hits'] = gt_df['hits'].apply(eval)
evaluator = rag.AIDTEvaluator(gt_df)
query_metrics_dict = evaluator.get_metrics(output_rankings_df)
metrics_dict = rag.AIDTEvaluator.get_metrics_by_type(query_metrics_dict)
# print(json.dumps(metrics_dict, indent=4))
metrics_df = rag.AIDTEvaluator.get_metrics_as_dataframe(query_metrics_dict, who=selector)
#print(metrics_df) # This dataframe is useful for generating the boxplots
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print(metrics_df)