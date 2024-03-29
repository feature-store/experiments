data_dir=/data/devangjhabakh/wikipedia/wikipedia

# download diffs
mkdir -p ${data_dir}/diffs;
aws s3 sync s3://feature-store-datasets/wikipedia/diffs ${data_dir}/diffs; 

# download model 
aws s3 cp s3://feature-store-datasets/wikipedia/models/bert-base-encoder.cp ${data_dir};

# download questions 
aws s3 cp s3://feature-store-datasets/wikipedia/10062021_filtered_questions.csv ${data_dir};

# download embeddings
mkdir -p ${data_dir}/embeddings; 
aws s3 sync s3://feature-store-datasets/wikipedia/embeddings ${data_dir}/embeddings;

## download raw api data
mkdir -p ${data_dir}/recentchanges; 
mkdir -p ${data_dir}/doc_xml; 
aws s3 sync s3://feature-store-datasets/wikipedia/recentchanges ${data_dir}/recentchanges;
aws s3 sync s3://feature-store-datasets/wikipedia/doc_xml ${data_dir}/doc_xml;
