python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --query "What could potentially be the next step? select from options: 0 clean inner wall of container;1 add water into the volumetric flask to the tick line;2 prepare the filler;3 disinfect;4 put the hamster into the hamster cage." \
    --video-file "data/COIN/videos/0/-8NaVGEccgc.mp4" \
    --num-video-frames 6 \
    --temperature 0
