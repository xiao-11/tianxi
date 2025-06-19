model_name=ui-tars

nohup python $DEBUG_MODE eval.py \
    --model_name $model_name \
    --gpus 0,1 \
    --output_path /data/lvbb1/ui-tars-result/screenQA \
    > /data/lvbb1/screen_qa-main/logs/0619.log 2>&1 &


# python $DEBUG_MODE eval_multhread.py \
#     --model_name $model_name \
#     --gpus 0,1 \
#     --output_path /data/lvbb1/ui-tars-result/screenQA



# python $DEBUG_MODE eval_mulprocessor.py \
#     --model_name $model_name \
#     --gpus 1 \
#     --output_path /data/lvbb1/ui-tars-result/screenQA