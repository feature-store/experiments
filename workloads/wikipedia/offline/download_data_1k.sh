data_dir=/data/jeffcheng1234/wikipedia/wikipedia

# download diffs
# mkdir -p ${data_dir}/diffs;
# aws s3 sync s3://feature-store-datasets/wikipedia/diffs ${data_dir}/diffs; 

#download edits
aws s3 cp s3://feature-store-datasets/wikipedia/edits_1k.csv ${data_dir}
aws s3 cp s3://feature-store-datasets/wikipedia/changes_1k.csv ${data_dir}
aws s3 cp s3://feature-store-datasets/wikipedia/top_titles_1k.csv ${data_dir}

# download model 
aws s3 cp s3://feature-store-datasets/wikipedia/models/bert-base-encoder.cp ${data_dir};

# download questions 
# aws s3 cp s3://feature-store-datasets/wikipedia/10062021_filtered_questions.csv ${data_dir};

# download embeddings
mkdir -p ${data_dir}/embeddings; 
# aws s3 sync s3://feature-store-datasets/wikipedia/embeddings ${data_dir}/embeddings;

## download raw api data
mkdir -p ${data_dir}/recentchanges; 
mkdir -p ${data_dir}/doc_xml; 
# aws s3 sync s3://feature-store-datasets/wikipedia/recentchanges ${data_dir}/recentchanges;
aws s3 sync s3://feature-store-datasets/wikipedia/new_doc_pkl/ ${data_dir}/new_doc_pkl;
