# Token and Head Adaptive Transformers for Efficient Natural Language Processing

Official PyTorch code for COLING 2022 paper:  [*Token and Head Adaptive Transformers for Efficient Natural Language Processing*](https://aclanthology.org/2022.coling-1.404/)  

The code is based on [HuggingFace's (🤗) Transformers](https://github.com/huggingface/transformers) library.

## Framework

![](./THAT.png)

## Dependencies:
+ Python 3.7.3
+ PyTorch 1.8.1
+ 🤗 Transformers
+ torchprofile

## Usage
### 1. Prepare data

Prepare GLUE dataset with `download_glue_data.py`

### 2. Finetune with Token and Head Drop

From a checkpoint finetuned with a downstream task, continue finetuning with Token and Head Drop.
```bash
python run_glue.py --model_name_or_path glue_output/$TASK_NAME/$MODEL_NAME/standard/checkpoint-best --task_name $TASK_NAME --do_train --do_eval --data_dir glue/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$TASK_NAME/$MODEL_NAME/joint_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.2 --layer_dropout_prob 0.2
```

### 3. Run Evolutionary Search of Joint Token and Head configuration

After training a Token and Head adaptive transformer, run an evolutionary search to find configurations with optimal accuracy-efficiency tradeoffs.
```bash
python run_glue.py --model_name_or_path glue_output/$TASK_NAME/$MODEL_NAME/joint_adaptive/checkpoint-best --task_name $TASK_NAME --do_search --do_eval --data_dir glue/$TASK_NAME --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output$TASK_NAME/$MODEL_NAME/evolutionary_search_joint --evo_iter 30 --mutation_size 30 --crossover_size 30
```



