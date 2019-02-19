import sys, nltk, operator
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(nltk.corpus.stopwords.words("english"))

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
    lmtzr = WordNetLemmatizer()

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
    lmtzr = WordNetLemmatizer()

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

    boqw = norm_question(question['text'])
    print(question['text'])
    print(norm_question(question['text']))

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
        
        answers.append((overlap, sent))
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    best_answer = (answers[0])[1]
    print(best_answer)    
    return best_answer



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
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
