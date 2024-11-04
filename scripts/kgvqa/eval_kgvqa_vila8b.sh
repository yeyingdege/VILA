CUDA_VISIBLE_DEVICES=0 python -m inference_test.evaluate_kg_video \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix \
    --conv-mode llama_3 \
    --image-folder data/COIN/videos \
    --question-file data/testing_vqa19_25oct_v2_pred_top5.json \
    --answers-file data/answers_25oct/answers_vila8b_f8_25oct_pred_top5.json \
    --max_new_tokens 512 \
    --num_video_frames 8 \
    --temperature 0
