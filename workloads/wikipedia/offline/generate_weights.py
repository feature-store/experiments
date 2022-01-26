
def generate_question_weights(questions_file, weights_dir, cutoff=None):
    filename = f"{weights_dir}/weights_{cutoff}.json"
for key in weights: 
    w = int(weights[key]/10)
    if w == 0: 
        w = 1
    for b in buckets: 
        if w <= b:
            w = b
            break
    weights[key] = b
    #print(weights[key], b)

    open(filename, "w").write(json.dumps(weights))

def generate_pageview_weights(pageview_file, weights_dir): 

    # generate weights by question




    # generate weights by pageview
    pageview_df = pd.read_csv(pageview_file)
    total_views = pageview_df.iloc[:, 2:].sum(axis=1).sum()
    weights = pageview_df.iloc[:, 2:].sum(axis=1) / total_views 
    pageview_df['weights'] = weights
    pageview_df.to_csv(pageview_file)

    # page weights per timestamp
    ts_to_weights = {}
    dates = pageview_df.columns[2:-2]
    for date in dates: 
        print(date)
        dt = datetime.strptime(date[:-2], '%Y%m%d')
        ts = dt.timestamp() * 1000000000
        ts_min = assign_timestamps_min(ts)
        view_counts = pageview_df[date].tolist()
        id_to_count  = pageview_df.set_index("doc_id")[date].to_dict()
        ts_to_weights[ts_min] = id_to_count
    open(timestamp_weights_file, "w").write(json.dumps(ts_to_weights))
    print("Generated ts weights file", timestamp_weights_file)




