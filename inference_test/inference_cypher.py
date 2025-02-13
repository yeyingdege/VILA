import argparse
import torch
import os
import re
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from llava.model.builder import load_pretrained_model
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


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_TaskSameToolSamePurpose', 'qa19_TaskSameObjectSamePurpose']


def main(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
        model_name, model_base=args.model_base
        )

    #model.config.tokenizer_model_max_length = args.tokenizer_model_max_length

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

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
                n_images = len(images)

                images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

                # replace <video> with <image>\n<image>\n....
                img_placehoder = '<image>\n' * n_images 
                if "<video>\n" in qs:
                    qs = qs.replace("<video>\n", img_placehoder)
                else:
                    qs = img_placehoder + qs
            else: # blind qa, don't use <image> in prompt
                qs = qs.replace("<video>\n", "")
            # qs = qs.replace("- (0)", "select from options: (0)")

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            if args.num_video_frames > 0:
                input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                output_ids = model.generate(
                    input_ids,
                    images=[images_tensor],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            else:
                inputs = tokenizer([prompt])
                input_ids = torch.as_tensor(inputs.input_ids).cuda()
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            print("input_ids shape", input_ids.shape[1])
        total += 1
        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print(outputs)
        # answer_id = parse_choice(outputs, line["all_choices"], line["index2ans"])
        results[idx]["response"] = outputs
        # results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(outputs, answer_id, gt_answers))


    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/cypher/cypher_prompt_with_example_test.json")
    parser.add_argument("--answers-file", type=str, default="data/cypher/answers_cypher/cypher_with_example_test_vila8b_blind.json")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_video_frames", type=int, default=0)
    #parser.add_argument("--tokenizer_model_max_length", type=int, default=8192)
    args = parser.parse_args()
    main(args)
