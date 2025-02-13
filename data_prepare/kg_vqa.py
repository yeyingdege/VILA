import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from process_retrievel import get_retrieval_info, get_task_step, std_taskname, get_cypher_retrieval_info


CONFIG = {
    "task_instructions": "",
    "multi_choice_example_format": """{} select one from options:
{}
Return only the index of the correct answer (e.g. 1, 2, 3, 4, or 5)."""
}

CONFIG2 = {
    "task_instructions": "The multiple-choice question is based on an instructional video. The goal is to select the best possible option based on the question and given choices, using reasoning and knowledge.\n",
    "multi_choice_example_format": """{}
{}
Return only the index of the correct answer (e.g. 1, 2, 3, 4, or 5)."""
}

CONFIG_FOR_PRED = {
    "task_instructions": "",
    "multi_choice_example_format": """A vision model's prediciton results of task recognition and step recognition is provided below.
Top {} task predictions:
{}
Top {} step predictions:
{}
{} select one from options:
{}
Return only the index of the correct answer (e.g. 1, 2, 3, 4, or 5)."""
}

CONFIG_FOR_RETRIEVAL = {
    "task_instructions": "",
    "multi_choice_example_format": """You are given the predicted task and step, their confident score, and the information retrieved from a knowledge graph based on the question. Answer the question based on the provided information.
{}
{} select one from options:
{}
Return only the index of the correct answer (e.g. 1, 2, 3, 4, or 5)."""
}

CONFIG_FOR_CYPHER_RETRIEVAL = {
    "task_instructions": "",
    "multi_choice_example_format": """You are given the predicted task and step, their confident score, and the information retrieved from a knowledge graph using the cypher queries. Answer the question based on the provided information.
{}
{} select one from options:
{}
Return only the index of the correct answer (e.g. 1, 2, 3, 4, or 5)."""
}


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


def split_words(input_string):
    if " " in input_string:
        return input_string
    formatted_string = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', input_string)
    return formatted_string
# print(split_words("MakeRJ45Cable"))

# to_replace = ["AssembleDesktopPC", "ReplaceBatteryOnTVControl", "MakeRJ45Cable", 
#               "AttendNBASkillsChallenge", "PerformCPR", "ReplaceCDDriveWithSSD",
#               "ReplaceSIMCard"]
replacement = {"Assemble Desktop P C": "Assemble Desktop PC",
               "Replace Battery On T V Control": "Replace Battery On TV Control",
               "Make R J45 Cable": "Make RJ45 Cable",
               "Make R J45Cable": "Make RJ45 Cable",
               "Attend N B A Skills Challenge": "Attend NBA Skills Challenge",
               "Perform C P R": "Perform CPR",
               "Replace C D Drive With S S D": "Replace CD Drive With SSD",
               "Replace S I M Card": "Replace SIM Card"}


def form_options(options, replacement):
    opts = ""
    all_choices = []
    index2ans = {}
    for ii, one_opt in enumerate(options):
        one_opt = split_words(one_opt)
        if one_opt in replacement.keys():
            one_opt = replacement[one_opt]
        opts += ("({}) {}\n".format(ii+1, one_opt))
        all_choices.append(str(ii+1))
        index2ans[str(ii+1)] = one_opt
    opts = opts.rstrip("\n")
    return opts, all_choices, index2ans


def form_task_step_preds(pred, topk=5):
    task_top5_classes = pred["task_top5_classes"]
    task_top5_scores = pred["task_top5_scores"]
    step_top5_classes = pred["step_top5_classes"]
    step_top5_scores = pred["step_top5_scores"]

    if topk > 1:
        task_string = ""
        step_string = ""
        # normalize scores
        sum1 = sum(task_top5_scores[:topk])
        task_top5_scores = [round(score / sum1, 4) for score in task_top5_scores]
        sum2 = sum(step_top5_scores[:topk])
        step_top5_scores = [round(score / sum2, 4) for score in step_top5_scores]
        for i in range(topk):
            curr_task = f"{std_taskname(task_top5_classes[i])} ({task_top5_scores[i]}), "
            task_string = task_string + curr_task
            curr_step = f"{step_top5_classes[i]} ({step_top5_scores[i]}), "
            step_string = step_string + curr_step
        task_string = task_string.rstrip(", ")
        step_string = step_string.rstrip(", ")
    else:
        task_string = std_taskname(task_top5_classes[0])
        step_string = step_top5_classes[0]
    return task_string, step_string


def load_json(one_file, miss_vid_file, video_dir, 
              use_pred_in_prompt, pred_file, 
              use_retrieval, retrieval_file, topk=5):
    with open(miss_vid_file, "r") as f:
        lines = f.readlines()
        miss_list = [a.strip() for a in lines]

    annots = ""
    with open(one_file, "r") as f:
        annots = json.load(f)

    if use_pred_in_prompt:
        preds = json.load(open(pred_file, "r"))
        pred_dict = {list(item.keys())[0]: list(item.values())[0] for item in preds}
    
    if use_retrieval:
        retrieval_infos = json.load(open(retrieval_file, "r"))
        preds = json.load(open(pred_file, "r"))
        pred_dict = {list(item.keys())[0]: list(item.values())[0] for item in preds}

    sft_annots = []
    none_retrieval_cnt = 0
    for one_line in tqdm(annots):
        video_id = one_line['video_id']
        if video_id in miss_list:
            print("Video file is missiong. {}".format(video_id))
            continue

        question = one_line['question']
        options  = one_line['options']
        answer   = one_line['answer'] + 1
        step_id  = one_line['step']['id']
        start_secs = one_line['step']['segment'][0]
        end_secs   = one_line['step']['segment'][1]
        question_type = one_line['quest_type']

        one_sample = {}
        one_sample["qid"] = one_line['qid']
        one_sample["video"] = find_video_path_by_id(video_id, video_dir)

        opts, all_choices, index2ans = form_options(options, replacement)
        if use_pred_in_prompt:
            try:
                pred = pred_dict[one_line['qid']].copy()
                task_string, step_string = form_task_step_preds(pred, topk=topk)
                question = CONFIG_FOR_PRED["task_instructions"] + "<video>\n" \
                    + CONFIG_FOR_PRED["multi_choice_example_format"] \
                    .format(topk, task_string, topk, step_string, question, opts)
            except KeyError:
                continue
        elif use_retrieval:
            pred = pred_dict[one_line['qid']].copy()
            if one_line['qid'] in retrieval_infos:
                retrieval_info = retrieval_infos[one_line['qid']]
            else:
                none_retrieval_cnt += 1
                retrieval_info = {}
            retrieval = get_cypher_retrieval_info(retrieval_info, pred)
            question = CONFIG_FOR_CYPHER_RETRIEVAL["task_instructions"] + "<video>\n" \
                    + CONFIG_FOR_CYPHER_RETRIEVAL["multi_choice_example_format"] \
                    .format(retrieval, question, opts)
        else:
            # question = "<video>\n{} select from options: {}.".format(question, opts)
            question = CONFIG["task_instructions"] + "<video>\n" + CONFIG["multi_choice_example_format"].format(question, opts)
        # print(question)

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
    print("none_retrieval_cnt", none_retrieval_cnt)
    return sft_annots


def main(args):
    kgvqa_dir = args.kgvqa_dir
    miss_vid_file = os.path.join(kgvqa_dir, args.miss_vid_file)

    json_path = args.in_file
    print("Processing {}...".format(json_path))
    sft_annos = load_json(json_path, miss_vid_file, args.video_dir, 
                            args.use_pred_in_prompt, args.pred_file, 
                            args.use_retrieval, args.retrieval_file, args.topk)

    with open(args.out_file, "w") as f:
        json.dump(sft_annos, f, indent=2)
    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-vid-file", type=str, default="miss_vid_list_1494.txt")
    parser.add_argument("--kgvqa-dir", type=str, default="data/kgvqa")
    parser.add_argument("--video-dir", type=str, default="data/COIN/videos")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--use_pred_in_prompt", type=bool, default=False)
    parser.add_argument("--pred_file", type=str, default="data/QA_25Oct24_testing_pred.json")
    parser.add_argument("--use_retrieval", type=bool, default=False)
    parser.add_argument("--retrieval_file", type=str, default="data/rephrased_QA_25Oct24_v2/cypher_with_example_test_10_vila8b_blind-retrieved_data.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--in-file", type=str, default="data/kgvqa/testing.json")
    parser.add_argument("--out-file", type=str, default="data/testing_vqa17_12Feb25.json")
    args = parser.parse_args()
    main(args)
