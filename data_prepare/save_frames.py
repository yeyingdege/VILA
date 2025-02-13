import os
import json
from PIL import Image
from llava.data.decord_func import decord_video_given_start_end_seconds


target_qid = "qa15_ToolOtherPurpose_19354"
result_dir = "output/sample_frames/"
figure_dir = result_dir + target_qid
os.makedirs(figure_dir, exist_ok=True)

data_file = "data/testing_vqa19_25oct_v2.json"
image_folder = "data/COIN/videos"
data = json.load(open(data_file, "r"))

for line in data:
    qid = line["qid"]
    if qid != target_qid:
        continue
    video_path = os.path.join(image_folder, line["video"])
    start_secs = line['start_secs']
    end_secs = line['end_secs']
    frames, frame_indices = decord_video_given_start_end_seconds(video_path, 
                            start_secs=start_secs, end_secs=end_secs,
                            num_video_frames=8)
    images =[ Image.fromarray(x).convert('RGB') for x in frames ]
    for i, image in enumerate(images):
        filename = os.path.join(figure_dir, str(i)+".png")
        image.save(filename)
        print("Saved figure to", filename)
