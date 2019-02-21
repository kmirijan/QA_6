import re, sys, nltk, operator
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(nltk.corpus.stopwords.words("english"))
lmtzr = WordNetLemmatizer()

# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at"])



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
    answers = []

    print("story")
    print(story)
    
    #question is an object
    print("Question: ", question["text"])
    boqw = norm_question(question['text'])
    qgraph = question["dep"]
    qmain = find_main(qgraph)
    #root word
    #AKA verb
    qword = qmain["word"]
    print("QWORD")
    print(qword)    
    
    stem = lmtzr.lemmatize(qword, wordnet.VERB)
    boqw.add(qword)

    print("-----STORY------")
    print(story['text'])

    text = ''
    if question['type'] == "sch":
        print("Question type is sch")
        text = story['sch']
#        print(text)
        sgraph = story["sch_dep"][1]
    else:
        print("Question type is Story")
        text = story['text']
#        print(text)
        sgraph = story["story_dep"][1]

    print(sgraph)
    snode = find_node(qword, sgraph)
    print("snode")
    print(snode)

    for node in sgraph.nodes.values():
        print("node[head]=", node["head"])
        if node.get('head', None) == snode["address"]:
            print("word, relation")
            print(node["word"], node["rel"])
            """
            if node['rel'] == "nmod":
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)
            """



    """
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        # A list of all the word tokens in the sentence
        bosw = norm_text(sent)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(boqw & bosw)
        answers.append((overlap, sent, bosw))
        
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    

    best_answer = (answers[0])[1]
    second_best_answer = (answers[1])[1]
    bosw = (answers[0])[2]
    print("bosw")
    print(bosw)
    print("boqw")
    print(boqw)
#    print("Best answer: ", best_answer)
#    print("Second best answer: ",second_best_answer)

    #get overlap values for top two answers
    max_overlap = answers[0][0]
    second_overlap = answers[1][0]

    
    #assuming if top two have same overlap, pick second answerr
    if max_overlap == second_overlap:
        print("---Top two have same overlap----")
        best_answer = second_best_answer

    print('\n')    
    return best_answer
    """


#Finds the verb
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None


def find_node(word, graph):
    print("Find node of --", word)
    for node in graph.nodes.values():
        if node["word"] == word:
            print("Found {} at {}".format(word, node))
            return node
    return None


def sent_test():
    print("In sent_test()")
    driver = QABase()
    q = driver.get_question("mc500.train.0.10")
    story = driver.get_story(q["sid"])

    #get the dependency graph of the firrst question
    qgraph = q["dep"]
    #print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo

    #sch_dep is dependency parses of Scherherrazade
    #sgraph = story["sch_dep"][1]
    story_graphs = story["story_dep"]

    #goes to find the verb
    qmain = find_main(qgraph)
    qword = qmain["word"]
    qpar = q['parse']
    print(qpar)
    qset = set()
    for node in qgraph.nodes.values():
        if node['word'] not in stopwords or node['word'] == qword:
            qset.add(node['lemma'])
    print(qset)
    #print(len(qgraph))



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
    #run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    #score_answers()

if __name__ == "__main__":
    main()
