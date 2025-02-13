import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from tqdm import tqdm


QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_TaskSameToolSamePurpose', 'qa19_TaskSameObjectSamePurpose']

grouping = {
    "tool": ["qa1_step2tool", "qa8_toolNextStep", "qa17_AlternativeTool"],
    "step": ["qa2_bestNextStep", "qa3_nextStep", "qa6_precedingStep", "qa7_bestPrecedingStep",
             "qa9_bestInitial", "qa10_bestFinal"],
    "task": ["qa18_TaskSameToolSamePurpose", "qa19_TaskSameObjectSamePurpose"],
    "domain": ["qa11_domain"],
    "purpose": ["qa12_toolPurpose", "qa13_actionPurpose", "qa14_objectPurpose",
                "qa15_ToolOtherPurpose", "qa16_ObjectOtherPurpose"]
}

def plot_wordcloud(words, save_name="test.jpg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=1600, height=800,
                background_color='white',
                stopwords=stopwords,
                min_font_size=10).generate(words)
 
    # plot the WordCloud image                       
    plt.figure(figsize=(16, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_name)


def get_words_from_question_type(data, question_types:list):
    """Create two word clouds given question types:
    1) word cloud of correct answers
    2) word cloud of all options"""
    pos_words, words = "", ""
    for line in tqdm(data, total=len(data)):
        # qid = line["qid"]
        quest_type = line["quest_type"]
        if quest_type in question_types:
            options = line["options"]
            answer = options[line["answer"]]
            new_options = []
            new_answer = answer.replace(" ", "_")
            # concat phrase with _
            for opt in options:
                new_options.append(opt.replace(" ", "_"))

            pos_words = pos_words + new_answer + " "
            words = words + " ".join(new_options) + " "
    pos_words = pos_words.lower().rstrip()
    words = words.lower().rstrip()
    return pos_words, words


if __name__ == "__main__":
    data_path = "data/rephrased_QA_25Oct24_v2/testing.json"
    data = json.load(open(data_path, "r"))
    key = "tool"
    pos_words, words = get_words_from_question_type(data, question_types=grouping[key])
    # pos_words, words = get_words_from_question_type(data, question_types=[key])
    plot_wordcloud(pos_words, save_name=f"output/wordcloud/positive_{key}_v1.jpg")
    plot_wordcloud(words, save_name=f"output/wordcloud/{key}_v1.jpg")
