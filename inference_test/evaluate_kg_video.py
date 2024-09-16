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
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, is_gemma_tokenizer, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from llava.model.builder import load_pretrained_model
from llava.data.decord_func import decord_video_given_start_end_seconds
from llava.data.dataset import LazySupervisedDataset
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

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa1_acc = TypeAccuracy("qa1_")
    qa2_acc = TypeAccuracy("qa2_")
    qa3_acc = TypeAccuracy("qa3_")
    qa4_acc = TypeAccuracy("qa4_")
    qa5_acc = TypeAccuracy("qa5_")
    qa6_acc = TypeAccuracy("qa6_")
    qa7_acc = TypeAccuracy("qa7_")
    qa8_acc = TypeAccuracy("qa8_")
    qa9_acc = TypeAccuracy("qa9_")
    qa10_acc = TypeAccuracy("qa10_")
    qa11_acc = TypeAccuracy("qa11_")


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

                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)

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
                    images=images_tensor,
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
        total += 1
        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        answer_id = parse_choice(outputs, line["all_choices"], line["index2ans"])
        results[idx]["response"] = outputs
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(outputs, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        if "qa1_" in quest_type:
            qa1_acc.update(gt_answers, answer_id)
        elif "qa2_" in quest_type:
            qa2_acc.update(gt_answers, answer_id)
        elif "qa3_" in quest_type:
            qa3_acc.update(gt_answers, answer_id)
        elif "qa4_" in quest_type:
            qa4_acc.update(gt_answers, answer_id)
        elif "qa5_" in quest_type:
            qa5_acc.update(gt_answers, answer_id)
        elif "qa6_" in quest_type:
            qa6_acc.update(gt_answers, answer_id)
        elif "qa7_" in quest_type:
            qa7_acc.update(gt_answers, answer_id)
        elif "qa8_" in quest_type:
            qa8_acc.update(gt_answers, answer_id)
        elif "qa9_" in quest_type:
            qa9_acc.update(gt_answers, answer_id)
        elif "qa10_" in quest_type:
            qa10_acc.update(gt_answers, answer_id)
        elif "qa11_" in quest_type:
            qa11_acc.update(gt_answers, answer_id)
        else:
            print(f"Unknown Type: {idx}")
        # print each type accuracy
        print("-----"*5)
        qa1_acc.print_accuracy()
        qa2_acc.print_accuracy()
        qa3_acc.print_accuracy()
        qa4_acc.print_accuracy()
        qa5_acc.print_accuracy()
        qa6_acc.print_accuracy()
        qa7_acc.print_accuracy()
        qa8_acc.print_accuracy()
        qa9_acc.print_accuracy()
        qa10_acc.print_accuracy()
        qa11_acc.print_accuracy()
        global_acc.print_accuracy()
        print("-----"*5)
        # average over type
        avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() 
                   + qa3_acc.get_accuracy() + qa4_acc.get_accuracy() 
                   + qa5_acc.get_accuracy() + qa6_acc.get_accuracy() 
                   + qa7_acc.get_accuracy() + qa8_acc.get_accuracy() 
                   + qa9_acc.get_accuracy() + qa10_acc.get_accuracy()
                   + qa11_acc.get_accuracy()) / 11.0
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa.json")
    parser.add_argument("--answers-file", type=str, default="data/answers.json")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_video_frames", type=int, default=6)
    #parser.add_argument("--tokenizer_model_max_length", type=int, default=8192)
    args = parser.parse_args()
    main(args)
