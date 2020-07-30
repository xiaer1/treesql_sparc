## An Interactive NL2SQL Approach With Reuse Strategy (The More You Ask, The Better You Query)

### Data
Downlaod data from [SParC](https://yale-lily.github.io/sparc). And then put in `sparc/` folder.
- What is SParC?
SParC is a dataset for cross-domain Semantic Parsing in Context. It is the context-dependent/multi-turn version of the Spider task, a complex and cross-domain text-to-SQL challenge. SParC consists of 4,298 coherent question sequences (12k+ unique individual questions annotated with SQL queries annotated by 14 Yale students), obtained from user interactions with 200 complex databases over 138 domains.

### Preporcessing data
Running the follow scripts, and then you will get preprocessed `pre_train.json`  and `pre_dev.json`.  
```
sh run_preprocess.sh path/to/save
```

### Train
Running the follow command:
```python
python3 train.py --cuda --epoch 21 \
--loss_epoch_threshold 10 \
--sketch_loss_coefficient 0.5 \
--column_pointer --use_query_attention --fine_tune_bert\
--action_embed_size 128 \
--type_embed_size 128 \
--maximum_utterances 6 \
--use_schema_encoder --use_copy_switch\
--encoder_state_size 300 \
--hidden_size 300 \
--encoder_num_layers 1 \


```
