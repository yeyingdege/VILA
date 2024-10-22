CUDA_VISIBLE_DEVICES=0 python -m inference_test.evaluate_kg_video \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --image-folder data/COIN/videos \
    --question-file data/testing_vqa20_rephrase_v2_pred_top3.json \
    --answers-file data/answers_rephrase_v2/answers_vila3b_f8_rephrase_v2_pred_top3.json \
    --max_new_tokens 512 \
    --num_video_frames 8 \
    --temperature 0
