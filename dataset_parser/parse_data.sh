python clone_kotlin_repositories.py $GITHUB_TOKEN /home/user/Kauto2/dataset_parser/kotlin_repositories
python extract_kotlin_code.py /home/user/Kauto2/dataset_parser/kotlin_repositories /home/user/Kauto2/dataset_parser/kotlin_files
python kotlin_code_to_parquet.py /home/user/Kauto2/dataset_parser/kotlin_files /home/user/Kauto2/dataset_parser/train_data/kotlin_train_data.parquet /home/user/Kauto2/dataset_parser/eval_data/kotlin_eval_data.parquet
