import json


num_question_per_type = 30
total_sample = num_question_per_type * 19

origin_file = "data/rephrased_QA_25Oct24_v2/testing.json"
save_file = "data/QA_25Oct24_testing_sampled.json"
miss_vid_file = "data/kgvqa/miss_vid_list_1494.txt"

sampled_json = {}
with open(origin_file, "r") as f:
    anns = json.load(f)

with open(miss_vid_file, "r") as f:
    lines = f.readlines()
    miss_list = [a.strip() for a in lines]

total_sample_cnt = 0
for ann in anns:
    if total_sample_cnt >= total_sample:
        break
    video_id = ann['video_id']
    if video_id in miss_list:
        print("Video file is missiong. {}".format(video_id))
        continue
    question_type = ann['quest_type']
    if question_type not in sampled_json:
        sampled_json[question_type] = []
    if len(sampled_json[question_type]) < num_question_per_type:
        sampled_json[question_type].append(ann)
        total_sample_cnt += 1
sample_data = []
for q, d in sampled_json.items():
    sample_data.extend(d)
with open(save_file, "w") as f:
    json.dump(sample_data, f, indent=2)
print("Saved sampled data to ", save_file)

