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


# retrieval_info = json.load(open("data/rephrased_QA_25Oct24_v2/retrieval_25Oct24_v2_2-testing.json", "r"))
# # test_exp = "qa12_toolPurpose_19191"
# # test_exp = "qa18_TaskSameToolSamePurpose_17818"
# test_exp = "qa1_step2tool_33305"
# info = retrieval_info[test_exp]
# res = get_retrieval_info(info)
# print(len(res), res)

