CUDA_VISIBLE_DEVICES=0 python -m inference_test.evaluate_kg_video \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --image-folder data/COIN/videos \
    --question-file data/testing_vqa_v2.json \
    --answers-file data/answers.json \
    --max_new_tokens 2048 \
    --num_video_frames 6 \
    --temperature 0
