import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def find_video_path_by_id(video_id, video_folder):
    # List all subfolders in video_folder
    video_folder = Path(video_folder)
    subfolders = [f.name for f in video_folder.iterdir() if f.is_dir()]

    # Iteratre subfolders, and find the video under subfolder
    for subfolder in subfolders:
        # Iterate all files in subfolder
        for file in (video_folder / subfolder).rglob('*'):
            #print(file, video_id)
            if video_id in file.name:
                return str(subfolder / Path(file.name))
    return "None"


def get_file_name(one_type, folder):
    file_name = ""
    pattern_to_match = "qa{}_".format(one_type)
    
    # list all json file under folder
    folder = Path(folder)
    json_files = list(folder.glob("*.json"))
    
    for json in json_files:
        json = str(json)
        if pattern_to_match in json:
            file_name = json     
    return file_name


def add_spaces_to_camel_case(text):
    # Check if the input already contains spaces
    if " " in text:
        return text
    # Add a space before each uppercase letter (except the first one) and join them
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
 

def load_json(one_file, miss_vid_file, video_dir):
    with open(miss_vid_file, "r") as f:
        lines = f.readlines()
        miss_list = [a.strip() for a in lines]

    annots = ""
    with open(one_file, "r") as f:
        annots = json.load(f)

    sft_annots = []
    for one_line in tqdm(annots):
        video_id = one_line['video_id']
        if video_id in miss_list:
            print("Video file is missiong. {}".format(video_id))
            continue

        question = one_line['question']
        options  = one_line['options']
        answer   = one_line['answer']
        step_id  = one_line['step']['id']
        start_secs = one_line['step']['segment'][0]
        end_secs   = one_line['step']['segment'][1]
        question_type = one_line['quest_type']

        one_sample = {}
        one_sample["qid"] = one_line['qid']
        one_sample["video"] = find_video_path_by_id(video_id, video_dir)

        opts = ""
        all_choices = []
        index2ans = {}
        for ii, one_opt in enumerate(options):
            if "qa5_task" in question_type:
                one_opt = add_spaces_to_camel_case(one_opt).lower()
            opts += ("({}) {};".format(ii, one_opt))
            all_choices.append(str(ii))
            index2ans[str(ii)] = one_opt
        opts = opts.rstrip(";")
        question = "<video>\n{} select from options: {}.".format(question, opts)

        one_sample["conversations"] = [
            { "from": "human", "value": question },
            { "from":"gpt", "value": "{} {}".format(answer, index2ans[str(answer)]) }
        ]
        one_sample["quest_type"] = question_type
        one_sample["start_secs"] = start_secs
        one_sample["end_secs"]   = end_secs
        one_sample["all_choices"] = all_choices
        one_sample["index2ans"]  = index2ans
        one_sample["task_label"] = one_line["task_label"]
        one_sample["step_label"] = one_line["step"]["label"]
        sft_annots.append(one_sample)

    return sft_annots


def main(args):
    kgvqa_dir = args.kgvqa_dir
    miss_vid_file = os.path.join(kgvqa_dir, args.miss_vid_file)

    for filename in os.listdir(kgvqa_dir):
        if filename.endswith('.json'):
            split, ext = filename.split(".")
            # if split is provided, only process the split file
            if args.split != "" and args.split not in split:
                continue

            json_path = os.path.join(kgvqa_dir, filename)
            print("Processing {}...".format(json_path))
            sft_annos = load_json(json_path, miss_vid_file, args.video_dir)

            sft_blindqa = os.path.join(args.out_dir, f"{split}_vqa.json")
            with open(sft_blindqa, "w") as f:
                json.dump(sft_annos, f, indent=2)
    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-vid-file", type=str, default="miss_vid_list_1494.txt")
    parser.add_argument("--kgvqa-dir", type=str, default="data/kgvqa")
    parser.add_argument("--video-dir", type=str, default="data/COIN/videos")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="")
    args = parser.parse_args()
    main(args)
