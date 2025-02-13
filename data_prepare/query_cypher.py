import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from kg_vqa import replacement, find_video_path_by_id, form_options


example = '''\
Example:
```
Top 5 task predictions: UnclogSinkWithBakingSoda (0.62), CleanBathtub (0.29), DrillHole (0.04), InstallShowerHead (0.03), MakeSlimeWithGlue (0.02)
Top 5 step predictions: clean bathtub with water (0.44), add baking soda to the sink hole (0.25), add hot water to the sink hole (0.12), apply detergent to bathtub (0.11), clean toys and hamster cages (0.08)
Question: What tool is suitable for this step?
Return only the python list of CYPHER queries.
CYPHER queries: [\"MATCH (t:Task)-[r:HAS_STEP]->(n:Step)-[g:HAS_GROUNDED_TOOL]->(m:GroundedTool) WHERE t.name='UnclogSinkWithBakingSoda' and n.name = 'add baking soda to the sink hole' return m\", \"MATCH (t:Task)-[r:HAS_STEP]->(n:Step)-[g:HAS_GROUNDED_TOOL]->(m:GroundedTool) WHERE t.name='UnclogSinkWithBakingSoda' and n.name = 'add hot water to the sink hole' return m\", \"MATCH (t:Task)-[r:HAS_STEP]->(n:Step)-[g:HAS_GROUNDED_TOOL]->(m:GroundedTool) WHERE t.name='CleanBathtub' and n.name = 'clean bathtub with water' return m\", \"MATCH (t:Task)-[r:HAS_STEP]->(n:Step)-[g:HAS_GROUNDED_TOOL]->(m:GroundedTool) WHERE t.name='CleanBathtub' and n.name = 'apply detergent to bathtub' return m\"]
```
'''

prompt='''\
You are given the schema of a knowledge graph stored in Neo4j. Generate a list of CYPHER queries to retrieve information that can be used to answer the provided question. 
The question is based on an instructional video, you are given the top five predictions of tasks and steps with their confident scores, predicted based on the video. 
Schema: 
Node properties:
Domain {{name: STRING}}
Task {{name: STRING, taskid: INTEGER}}
Step {{name: STRING, stepid: INTEGER}}
Action {{name: STRING}}
START {{name: STRING}}
END {{name: STRING}}
Object {{name: STRING}}
GroundedTool {{name: STRING}}
Purpose {{name: STRING}}
Relationship properties:
HAS_NEXT_STEP {{tasks: LIST, vids: LIST, freq: INTEGER}}
HAS_GROUNDED_TOOL {{vids: LIST, freq: INTEGER}}
HAS_SIMILAR_PURPOSE {{sim: FLOAT}}
The relationships:
(:Domain)-[:HAS_TASK]->(:Task)
(:Task)-[:HAS_STEP]->(:Step)
(:Task)-[:HAS_STEP]->(:END)
(:Task)-[:HAS_STEP]->(:START)
(:Step)-[:HAS_NEXT_STEP]->(:Step)
(:Step)-[:HAS_NEXT_STEP]->(:END)
(:Step)-[:HAS_ACTION]->(:Action)
(:Step)-[:HAS_GROUNDED_TOOL]->(:GroundedTool)
(:Step)-[:HAS_OBJECT]->(:Object)
(:Action)-[:HAS_PURPOSE]->(:Purpose)
(:START)-[:HAS_NEXT_STEP]->(:Step)
(:Object)-[:HAS_PURPOSE]->(:Purpose)
(:GroundedTool)-[:HAS_PURPOSE]->(:Purpose)
(:Purpose)-[:HAS_SIMILAR_PURPOSE]->(:Purpose)
{}
REAL INPUT: 

Top 5 task predictions: {}
Top 5 step predictions: {}

Question: {}
Return only the python list of CYPHER queries.
CYPHER queries:
'''

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
        task_top5_scores = [round(score / sum1, 2) for score in task_top5_scores]
        sum2 = sum(step_top5_scores[:topk])
        step_top5_scores = [round(score / sum2, 2) for score in step_top5_scores]
        for i in range(topk):
            curr_task = f"{task_top5_classes[i]} ({task_top5_scores[i]}), "
            task_string = task_string + curr_task
            curr_step = f"{step_top5_classes[i]} ({step_top5_scores[i]}), "
            step_string = step_string + curr_step
        task_string = task_string.rstrip(", ")
        step_string = step_string.rstrip(", ")
    else:
        task_string = task_top5_classes[0]
        step_string = step_top5_classes[0]
    return task_string, step_string


def load_json(one_file, miss_vid_file, video_dir, 
              use_example, pred_file, topk=5):
    with open(miss_vid_file, "r") as f:
        lines = f.readlines()
        miss_list = [a.strip() for a in lines]

    annots = ""
    with open(one_file, "r") as f:
        annots = json.load(f)

    preds = json.load(open(pred_file, "r"))
    pred_dict = {list(item.keys())[0]: list(item.values())[0] for item in preds}

    sft_annots = []
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

        pred = pred_dict[one_line['qid']].copy()
        task_string, step_string = form_task_step_preds(pred, topk=topk)

        if use_example:
            question = "<video>\n" + prompt.format(example, task_string, step_string, question)
        else:
            question = "<video>\n" + prompt.format("", task_string, step_string, question)
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

    return sft_annots


def main(args):
    kgvqa_dir = args.kgvqa_dir
    miss_vid_file = os.path.join(kgvqa_dir, args.miss_vid_file)

    json_path = args.in_file
    print("Processing {}...".format(json_path))
    annos = load_json(json_path, miss_vid_file, args.video_dir,
                      args.use_example, args.pred_file, args.topk)

    with open(args.out_file, "w") as f:
        json.dump(annos, f, indent=2)
    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--miss-vid-file", type=str, default="miss_vid_list_1494.txt")
    parser.add_argument("--kgvqa-dir", type=str, default="data/kgvqa")
    parser.add_argument("--video-dir", type=str, default="data/COIN/videos")
    parser.add_argument("--in-file", type=str, default="data/rephrased_QA_25Oct24_v2/testing.json")
    parser.add_argument("--out-file", type=str, default="data/cypher/cypher_prompt_with_example_test.json")
    parser.add_argument("--use_example", type=bool, default=True)
    parser.add_argument("--pred_file", type=str, default="data/QA_25Oct24_testing_pred.json")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    main(args)

