#!/bin/bash
#SBATCH --job-name=masakhanews-experiments
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --time=168:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/masakhane-news/comErrorXLMRBase.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/masakhane-news/comOutputXLMRBase.txt

###########cluster information above this line
<<<<<<< HEAD
# module load python/3.9 cuda/10.2/cudnn/7.6
# source /home/mila/b/bonaventure.dossou/afrispeech/bin/activate
# pip install -r requirements.txt
# export PYTHONPATH=$PYTHONPATH:~/masakhane-news
export MAX_LENGTH=164
export BATCH_SIZE=32
=======
module load python/3.9 cuda/10.2/cudnn/7.6
source /home/mila/b/bonaventure.dossou/afrispeech/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/masakhane-news
export MAX_LENGTH=256
export BATCH_SIZE=64
>>>>>>> b23a1fb87f7ef6f9da27a6a44c24959ce44aee64
export NUM_EPOCHS=10
export SAVE_STEPS=5000

for model in bonadossou/afrolm_active_learning  castorini/afriberta_large castorini/afriberta_base castorini/afriberta_small
do
	export BERT_MODEL=${model}
	if [ ${model} == xlm-roberta-base ]
	then
		declare -a langs=(eng fra hau ibo lin pcm run swa yor orm sna)
	else
		declare -a langs=(amh eng fra hau ibo lin pcm run swa yor orm sna)
	fi
	for lang in "${langs[@]}"
	do
		# export OUTPUT_DIR=/content/drive/MyDrive/masakhane-news/results/${lang}_${model}
		export DATA_DIR=/content/masakhane-news/data/${lang}
		export LABELS_FILE=${DATA_DIR}/labels.txt
		export LANG=${lang}
<<<<<<< HEAD
		export OUTPUT_DIR=/content/drive/MyDrive/MasakaneNews/results/${lang}/${model}
		for seed in 1 2 3 4 5
=======
		export OUTPUT_DIR=/home/mila/b/bonaventure.dossou/masakhane-news/results/${lang}_${model}
		if [[ ${lang} == eng && ${model} == xlm-roberta-base ]]
		then
			declare -a seeds=(4 5)
		else
			declare -a seeds=(1 2 3 4 5)
		fi
		for seed in "${seeds[@]}"
>>>>>>> b23a1fb87f7ef6f9da27a6a44c24959ce44aee64
		do
			for header_style in 0 1
			do
				export HEADER_STYLE=${header_style}
				export SEED=${seed}
<<<<<<< HEAD
				export OUTPUT_FILE=/content/drive/MyDrive/MasakaneNews/results/${lang}/${model}/test_result_${seed}_${header_style}
				export OUTPUT_PREDICTION=/content/drive/MyDrive/MasakaneNews/results/${lang}/${model}/test_predictions_${seed}_${header_style}
=======
				export OUTPUT_FILE=${OUTPUT_DIR}/test_result_${lang}_${seed}_${header_style}
				export OUTPUT_PREDICTION=${OUTPUT_DIR}/test_predictions_${lang}_${seed}_${header_style}
>>>>>>> b23a1fb87f7ef6f9da27a6a44c24959ce44aee64

				CUDA_VISIBLE_DEVICES=0 python code/train_textclass.py --data_dir $DATA_DIR \
				--model_type xlmroberta \
				--model_name_or_path $BERT_MODEL \
				--output_dir $OUTPUT_DIR \
				--output_result $OUTPUT_FILE \
				--output_prediction_file $OUTPUT_PREDICTION \
				--max_seq_length  $MAX_LENGTH \
				--num_train_epochs $NUM_EPOCHS \
				--learning_rate 5e-5 \
				--per_gpu_train_batch_size $BATCH_SIZE \
				--per_gpu_eval_batch_size $BATCH_SIZE \
				--save_steps $SAVE_STEPS \
				--seed $SEED \
				--labels $LABELS_FILE \
				--save_total_limit 1 \
				--gradient_accumulation_steps 2 \
				--do_train \
				--do_eval \
				--do_predict \
				--overwrite_output_dir \
				--header $HEADER_STYLE
			done
		done
	done
done