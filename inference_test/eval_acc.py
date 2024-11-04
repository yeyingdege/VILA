import json
from evaluate_kg_video import TypeAccuracy, QUESTION_TYPES


# load answer file
answers = json.load(open("data/answers_25oct/answers_owl3_f8_25oct_pred_top3.json", "r"))


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
avg_acc = sum(acc_list) / len(acc_list)
print("Average Acc over Type: {:.4f}".format(avg_acc))
