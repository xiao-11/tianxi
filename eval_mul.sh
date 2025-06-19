model_name=ui-tars

# python $DEBUG_MODE eval.py \
#     --model_name $model_name \
#     --gpus 1 \
#     --output_path /data/lvbb1/ui-tars-result/screenQA

python $DEBUG_MODE eval_mulprocessor.py \
    --model_name $model_name \
    --gpus 1 \
    --output_path /data/lvbb1/ui-tars-result/screenQA