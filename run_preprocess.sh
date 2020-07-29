#!/bin/sh

folder="process_data"

if [ ! -d "$folder" ]; then
 mkdir "$folder"
fi

echo 'Satrt processing datasets with value'

python3 preprocess/data_process_sparc.py --data_path sparc/train.json --table_path sparc/tables.json --output $folder/schema_train.json

python3 preprocess/data_process_sparc.py --data_path sparc/dev.json --table_path sparc/tables.json --output $folder/schema_dev.json

echo 'Satrt convert sql2TreeSQL with value'

python3 preprocess/sql2SemQL.py --data_path $folder/schema_train.json --table_path sparc/tables.json --output $folder/pre_train.json

python3 preprocess/sql2SemQL.py --data_path $folder/schema_dev.json --table_path sparc/tables.json --output $folder/pre_dev.json

cp sparc/tables.json $folder
