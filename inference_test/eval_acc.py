import json
from evaluate_kg_video import TypeAccuracy, QUESTION_TYPES


def eval(answer_file="data/answers_25oct/answers_owl3_f8_25oct_pred_top5.json"):
    # load answer file
    answers = json.load(open(answer_file, "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))

    for qid, item in answers.items():
        quest_type = item["quest_type"]
        gt_answers = item["gt"]
        answer_id = item["parser"]

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
    avg_acc = sum(acc_list) / min(len(acc_list), 17)
    print("Average Acc over Type: {:.4f}".format(avg_acc))



def get_all_qids(file="data/kgvqa/testing.json"):
    data = json.load(open(file, "r"))
    qid_list = [d["qid"] for d in data if d["quest_type"] not in ['qa4_step','qa5_task']]
    return qid_list


def filter_results_by_qid(result, qid_file):
    qid_list = get_all_qids(qid_file)
    new_result = {qid: result[qid] for qid in qid_list}
    return new_result


def merge_results(file1, file2,
                  skip_qt=['qa4_step','qa5_task']):
    """file1: filenames with suffix "rephrase_v2" contains evaluation results of question type 1-11. 
            NOTE that qa8 is outdated in this version.
    file2: filenames with suffix "25oct" contains evaluation results of question type 8, 12-19.
    """
    data1 = json.load(open(file1, "r"))
    data2 = json.load(open(file2, "r"))
    merged_data = data2.copy()
    for qid, item in data1.items():
        if item["quest_type"] in skip_qt:
            continue
        # # don't use qa8 results of file1
        # if item["quest_type"]=='qa8_toolNextStep':
        #     continue
        merged_data[qid] = item
    return merged_data

def main_merge_and_filter():
    cvpr_ds_num = 48019
    iccv_ds_num = 46921
    qid_file = "data/kgvqa/testing.json"
    file1 = "data/answers_25oct_full/answers_vila8b_f8_25oct.json"
    file2 = "data/answers_12Feb25/answers_vila8b_f8_12Feb25.json"
    save_file="data/answers_12Feb25_full/answers_vila8b_f8_12Feb25.json"

    merged_data = merge_results(file1, file2)
    filtered_data = filter_results_by_qid(merged_data, qid_file)
    assert len(filtered_data)==iccv_ds_num, f"total number of test set is incorrect! {len(merged_data)} not 48019"
    with open(save_file, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print("Saved data to", save_file)


if __name__=="__main__":
    # main_merge_and_filter()

    answer_file = "data/answers_12Feb25_full/answers_vila8b_f8_12Feb25.json"
    eval(answer_file)
