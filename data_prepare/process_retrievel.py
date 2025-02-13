import json
import re


replacement = {"Assemble Desktop P C": "Assemble Desktop PC",
               "Replace Battery On T V Control": "Replace Battery On TV Control",
               "Make R J45 Cable": "Make RJ45 Cable",
               "Make R J45Cable": "Make RJ45 Cable",
               "Attend N B A Skills Challenge": "Attend NBA Skills Challenge",
               "Perform C P R": "Perform CPR",
               "Replace C D Drive With S S D": "Replace CD Drive With SSD",
               "Replace S I M Card": "Replace SIM Card"}

def std_taskname(itxt):
    tmp = re.findall('[A-Z][^A-Z]*', itxt)
    tmp = ' '.join(tmp).strip()
    return replacement.get(tmp, tmp)

def get_retrieval_text(rev):
    '''
     types of retrieval:
    {} 
    []
    [[name, freq], []]
    '''
    if len(rev) == 0:
        return ""
    if isinstance(rev, list):
        if isinstance(rev[0], list): 
            # list within list 
            tmp = []
            for a in rev:
                tmp.append(f"{a[0]} ({a[1]})")
            return f"{', '.join(tmp).strip()}"
        else:
            rev = list(set(rev))
            return f"{', '.join(rev).strip()}"
    elif isinstance(rev, dict):
        tmp1 = []
        for k, v in rev.items():
            v = list(set(v))
            tmp1.append(f"{k}: {', '.join(v).strip()}")
        return '; '.join(tmp1).strip()
    print("UNIDENTIFIED RETRIEVAL!")
    return ""


def get_retrieval_info_single(info):
    '''
    <Predicted Task (score)> <Predicted Step (Score)> <retrieved informtion> 
    '''
    r = info['retrieved_data']
    task_score = info['task_score']
    step_score = info['step_score']
    task_score = f"{task_score:.2f}" if isinstance(task_score, float) else task_score
    step_score = f"{step_score:.2f}" if isinstance(step_score, float) else step_score
    return f"<{std_taskname(info['task'])} ({task_score})> <{info['step']} ({step_score})> <{get_retrieval_text(r)}>" 

def get_retrieval_info(info):
    '''
    <Predicted Task (score)> <Predicted Step (Score)> <retrieved informtion> 
    '''
    tmp = []
    cypher = None 
    for d in info: 
        if cypher is None:
            cypher = d['cypher']
        tmp.append(get_retrieval_info_single(d))
    txt = ''
    txt = f"The sample query to retrieve the information is as follows: {cypher}\n"
    txt += "Retrieved information. Format: <Predicted Task (score)> <Predicted Step (Score)> <retrieved informtion>. \n"
    txt += '\n'.join(tmp)
    return txt


def get_task_step(pred):
    '''
    <Predicted Task (score)> <Predicted Step (Score)> <retrieved informtion> 
    '''
    task_top5_classes = pred["task_top5_classes"]
    task_top5_scores = pred["task_top5_scores"]
    step_top5_classes = pred["step_top5_classes"]
    step_top5_scores = pred["step_top5_scores"]
    tmp = []
    for task, step, task_score, step_score in zip(task_top5_classes, step_top5_classes, task_top5_scores, step_top5_scores):
        task_score = f"{task_score:.2f}" if isinstance(task_score, float) else task_score
        step_score = f"{step_score:.2f}" if isinstance(step_score, float) else step_score
        tmp.append(f"<{std_taskname(task)} ({task_score})> <{step} ({step_score})>")
    txt = ""
    txt += "Retrieved information. Format: <Predicted Task (score)> <Predicted Step (Score)>. \n"
    txt += '\n'.join(tmp)
    return txt


def get_task_step_compact(pred):
    '''
    Top 5 task predictions: str (), str()...
    Top 5 step predictions: str (), str()...
    '''
    task_top5_classes = pred["task_top5_classes"]
    task_top5_scores = pred["task_top5_scores"]
    step_top5_classes = pred["step_top5_classes"]
    step_top5_scores = pred["step_top5_scores"]

    task_string = ""
    step_string = ""
    # normalize scores
    sum1 = sum(task_top5_scores)
    task_top5_scores = [round(score / sum1, 2) for score in task_top5_scores]
    sum2 = sum(step_top5_scores)
    step_top5_scores = [round(score / sum2, 2) for score in step_top5_scores]
    for i in range(len(task_top5_classes)):
        curr_task = f"{std_taskname(task_top5_classes[i])} ({task_top5_scores[i]}), "
        task_string = task_string + curr_task
        curr_step = f"{step_top5_classes[i]} ({step_top5_scores[i]}), "
        step_string = step_string + curr_step
    task_string = task_string.rstrip(", ")
    step_string = step_string.rstrip(", ")
    txt = f"Top 5 task predictions: {task_string}\nTop 5 step predictions: {step_string}"
    return txt


def get_cypher_retrieval_info(info, pred):
    '''
    CYPHER query: ""
    Retrieved data: []
    '''
    pred_str = get_task_step_compact(pred)
    tmp = []
    for k, v in info.items():
        s = ""
        if v is not None and len(v) > 0:
            s += f"CYPHER query: {k}\nRetrieved data: {v}"
            tmp.append(s)
    if len(tmp) > 0:
        txt = "Retrieved data for the relevant CYPHER query:\n"
        txt += '\n'.join(tmp)
    else:
        txt = "Retrieved data for the relevant CYPHER query: None\n"
    txt = f"{pred_str}\n{txt}"
    return txt


# retrieval_info = json.load(open("data/rephrased_QA_25Oct24_v2/cypher_with_example_test_10_vila8b_f8-retrieved_data.json", "r"))
# preds = json.load(open("data/QA_25Oct24_testing_pred.json", "r"))
# pred_dict = {list(item.keys())[0]: list(item.values())[0] for item in preds}
# # test_exp = "qa12_toolPurpose_19191"
# # test_exp = "qa18_TaskSameToolSamePurpose_17818"
# # test_exp = "qa1_step2tool_33305"
# test_exp = "qa1_step2tool_36614" # without retrieval data
# info = retrieval_info[test_exp]
# pred = pred_dict[test_exp].copy()
# res = get_cypher_retrieval_info(info, pred)
# print(res)

