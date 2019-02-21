import sys, nltk, operator
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(nltk.corpus.stopwords.words("english"))
lmtzr = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def norm_question(question):
    quest_words = nltk.word_tokenize(question)
    quest_words = nltk.pos_tag(quest_words)
    del quest_words[0]
    del quest_words[-1]

    root_question_words = set()
    for word_pair in quest_words:
        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag = get_wordnet_pos(word_pair[1])
            stem = lmtzr.lemmatize(word, tag)
            root_question_words.add(stem)

    return root_question_words

def norm_text(sent):
    sent_words = nltk.word_tokenize(sent)
    sent_words = nltk.pos_tag(sent_words)

    root_sent_words = set()
    for word_pair in sent_words:
        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag = get_wordnet_pos(word_pair[1])
            stem = lmtzr.lemmatize(word, tag)
            root_sent_words.add(stem)

    return root_sent_words


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###
    print("IN GET ANSWER")
    answers = []

    boqw = norm_question(question['text'])
    qgraph = question["dep"]
    qmain = find_main(qgraph)
    #root word
    qword = qmain["word"]
    print("-----ROOT WORD----")
    print(qword)
    stem = lmtzr.lemmatize(qword, wordnet.VERB)
    boqw.add(qword)
    print(question['text'])

    text = ''
    if question['type'] == "Story":
        text = story['text']
    else:
        text = story['sch']

    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        # A list of all the word tokens in the sentence
        bosw = norm_text(sent)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(boqw & bosw)

        print("overlap: ", overlap)
    
        answers.append((overlap, sent, bosw))
        
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    for i in answers:
        print(i)
    best_answer = (answers[0])[1]
    second_best_answer = (answers[1])[1]
    bosw = (answers[0])[2]
    print("Best answer: ", best_answer)
    print("Second best answer: ",second_best_answer)

    #get overlap values for top two answers
    max_overlap = answers[0][0]
    second_overlap = answers[1][0]

    
    #assuming if top two have same overlap, pick second answerr
    if max_overlap == second_overlap:
        print("---Top two have same overlap----")
        best_answer = second_best_answer

    print('\n')    
    return best_answer

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None

def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None

def sent_test():
    driver = QABase()
    q = driver.get_question("mc500.train.0.10")
    story = driver.get_story(q["sid"])

    qgraph = q["dep"]
    qmain = find_main(qgraph)
    qword = qmain["word"]
    qset = set()
    for node in qgraph.nodes.values():
        if node['word'] not in stopwords or node['word'] == qword:
            qset.add(node['lemma'])
    print(qset)
    #print(len(qgraph))
            

    print(qword)
    story_graphs = story["story_dep"]


#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

#############################################################


def main():
    sent_test()
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    #score_answers()

if __name__ == "__main__":
    main()
