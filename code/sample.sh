
# export MAX_LENGTH=164
# export BATCH_SIZE=32
# export PYTHONPATH=$PYTHONPATH:~/masakhane-news
export MAX_LENGTH=256
export BATCH_SIZE=8
export NUM_EPOCHS=10
export SAVE_STEPS=500000

for model in Atnafu/Ethio-xmlr-large-250k #bonadossou/afrolm_active_learning  castorini/afriberta_large castorini/afriberta_base castorini/afriberta_small
do
	export BERT_MODEL=${model}

	for lang in amh eng orm tir som 
	do
		# export OUTPUT_DIR=/content/drive/MyDrive/masakhane-news/results/${lang}_${model}
		export DATA_DIR=/content/drive/MyDrive/DS/News/${lang}
		export LABELS_FILE=${DATA_DIR}/labels.txt
		export LANG=${lang}
		export OUTPUT_DIR=/content/drive/MyDrive/MasakaneNews/results/${lang}_${model}
		for seed in 1 2 3 4 5
		do
			for header_style in 0
			do
				export HEADER_STYLE=${header_style}
				export SEED=${seed}
				# export OUTPUT_FILE=/content/drive/MyDrive/MasakaneNews/results/${lang}_${model}/test_result_${seed}_${header_style}
				# export OUTPUT_PREDICTION=/content/drive/MyDrive/MasakaneNews/results/${lang}_${model}/test_predictions_${seed}_${header_style}
    		export OUTPUT_FILE=${OUTPUT_DIR}/test_result_${lang}_${seed}_${header_style}
				export OUTPUT_PREDICTION=${OUTPUT_DIR}/test_predictions_${lang}_${seed}_${header_style}

				CUDA_VISIBLE_DEVICES=0 python code/train_textclass.py --data_dir $DATA_DIR \
				--model_type xlmroberta \
				--model_name_or_path $BERT_MODEL \
				--output_dir $OUTPUT_DIR \
				--output_result $OUTPUT_FILE \
				--output_prediction_file $OUTPUT_PREDICTION \
				--max_seq_length  $MAX_LENGTH \
				--num_train_epochs $NUM_EPOCHS \
				--learning_rate 1e-5 \
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