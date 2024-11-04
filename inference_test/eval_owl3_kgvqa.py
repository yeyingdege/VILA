import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from transformers import AutoModel, AutoTokenizer
from llava.data.decord_func import decord_video_given_start_end_seconds
from llava.eval.mmmu_utils.eval_utils import parse_choice



class TypeAccuracy(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "{}".format(pred) in gt:
            self.correct += 1

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        #print(f"{self.type_name} Accuracy: {self.get_accuracy()} | {self.correct}/{self.total}")
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                int(self.total)
            ))


QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_TaskSameToolSamePurpose', 'qa19_TaskSameObjectSamePurpose']


def main(args):
    # Load Model
    model_path = os.path.expanduser(args.model_path)
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device='cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = model.init_processor(tokenizer)

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))


    total = 0
    results = {}
    for line in tqdm(annotations, total=len(annotations)):
        # Q-A Pair
        idx = line["qid"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[idx] = {"qid": idx, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}
        if "<video>\n" in qs:
            qs = qs.replace("<video>\n", "")
        if "<image>\n" in qs:
            qs = qs.replace("<image>\n", "")

        with torch.inference_mode():
            if args.num_video_frames > 0:
                # Load Image
                video_path = os.path.join(args.image_folder, line["video"])

                if "start_secs" in line:
                    start_secs = line['start_secs']
                    end_secs = line['end_secs']
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=args.num_video_frames)
                else:
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path,
                        num_video_frames=args.num_video_frames)

                images =[ Image.fromarray(x).convert('RGB') for x in frames ]
                messages = [
                    {"role": "user", "content": f"""<|video|>{qs}"""},
                    {"role": "assistant", "content": ""}
                ]
                inputs = processor(messages, images=None, videos=[images])
            else: # blind qa, don't use <image> in prompt
                messages = [
                    {"role": "user", "content": f"""{qs}"""},
                    {"role": "assistant", "content": ""}
                ]
                inputs = processor(messages, images=None, videos=None)
            inputs.to('cuda')
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens':args.max_new_tokens,
                'decode_text':True,
            })
            outputs = model.generate(**inputs)[0]
        total += 1
        outputs = outputs.strip()

        answer_id = parse_choice(outputs, line["all_choices"], line["index2ans"])
        results[idx]["response"] = outputs
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(outputs, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        for t in range(len(QUESTION_TYPES)):
            if f"qa{t+1}_" in quest_type:
                qa_acc[t].update(gt_answers, answer_id)

        # print each type accuracy
        print("-----"*5)
        acc_list = []
        for t in range(len(QUESTION_TYPES)):
            qa_acc[t].print_accuracy()
            acc_list.append(qa_acc[t].get_accuracy())
        global_acc.print_accuracy()
        print("-----"*5)
        avg_acc = sum(acc_list) / len(acc_list)
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mPLUG/mPLUG-Owl3-7B-240728")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa20_rephrase_v2_pred_top3.json")
    parser.add_argument("--answers-file", type=str, default="data/answers_rephrase_v2/answers_owl3_f8_rephrase_v2_pred_top3.json")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_video_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)
