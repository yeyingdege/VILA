CUDA_VISIBLE_DEVICES=2 python -m inference_test.evaluate_kg_video \
    --model-path Efficient-Large-Model/VILA1.5-40b \
    --conv-mode hermes-2 \
    --image-folder data/COIN/videos \
    --question-file data/testing_vqa.json \
    --answers-file data/answers_vila40b_f8.json \
    --max_new_tokens 512 \
    --num_video_frames 8 \
    --temperature 0
