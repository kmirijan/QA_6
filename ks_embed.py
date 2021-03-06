import re, sys, nltk, operator
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from word2vec_extractor import Word2vecExtractor
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

stopwords = set(nltk.corpus.stopwords.words("english"))
lmtzr = WordNetLemmatizer()

DATA_DIR = "data"
#w2vecmodel = "data/glove-w2v.txt"
w2vecmodel = "data/glove.6B.300d.txt"
w2v = None

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "from","on", "at", "along","under", "around","near","to","in front of"])

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences    


def get_words(text):
    words_list = []
    for sent in nltk.sent_tokenize(text):
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sent)):
            if word not in stopwords and re.search(r'\w', word):
                words_list.append(word.lower())
    
    print("words_list")
    print(words_list)
    return words_list


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
    quest_words = nltk.word_tokenize(question['text'])
    quest_words = nltk.pos_tag(quest_words)
    
    del quest_words[0]
    del quest_words[-1]

    root_question_words = set()
    
    # Adds the root word to troot_question_words
    qgraph = question["dep"]
    qword = find_main(qgraph)['lemma']
    root_question_words.add(qword)
    
    for word_pair in quest_words:
        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag  = get_wordnet_pos(word_pair[1])
            
            #Added in morphy function and added stems to root_question_words
            stems = wordnet._morphy(word, tag)
            [root_question_words.add(stem) for stem in stems]
            if word_pair[1] == 'NNP':
                root_question_words.add(word)

    print("Root question:", root_question_words)
    print("Qword:", qword)
    return root_question_words, qword

def norm_text(sent):
    sent_words = nltk.word_tokenize(sent)
    sent_words = nltk.pos_tag(sent_words)

    root_sent_words = set()

    for word_pair in sent_words:
        
        # If the word is a Verb, then add the word to root_sent_words
        if (word_pair[1].startswith('V')):
            [root_sent_words.add(word) for word in wordnet._morphy(word_pair[0].lower(), wordnet.VERB)]
            continue

        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag  = get_wordnet_pos(word_pair[1])
            
            #Added in morphy function and added stems to root_question_words
            stems = wordnet._morphy(word, tag)
            [root_sent_words.add(stem) for stem in stems]
            if word_pair[1] == 'NNP':
                root_sent_words.add(word)

    return root_sent_words

def get_question_words(qgraph, qword):
    print("In get_question_words")
    qset = set()
    for node in qgraph.nodes.values():
        if node['word'] not in stopwords or node['word'] == qword:
            if node['address'] != 0 and node['address'] != 1 and node['rel'] != 'punct':
                qset.add(node['lemma'])
    print("qset: ", qset)
    return qset


def get_word_embedding_features(text):
    global w2v
    if w2v is None:
        print("loading word vectors ...", w2vecmodel)
        w2v = Word2vecExtractor(w2vecmodel)
    # might want to change this to sent2vec?
    #feature_dict = w2v.get_doc2vec_feature_dict(text)
    doc_model = w2v.doc2vec(text)
    sent_model = w2v.sen2vec(text)

    return sent_model

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)



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

    boqw, qword = norm_question(question)
    # print(question['text'])
    # print(boqw)
    # print(qword)

    text = ''
    
    if question['type'] == "Sch":
        text = story['sch']
        s_type = 'sch_dep'
    else:
        text = story['text']
        s_type = 'story_dep'
    
    sentences = nltk.sent_tokenize(text)
    #get tokenized list of words for story text
    #words = get_words(text)
   
    #word embeddings of the text is preprocessed in wor3vec_extractor
    #model = get_word_embedding_features(text)
    #print("model")
    #print(model)


    q_vector = get_word_embedding_features(question['text'])
#    print("sentences")
    q_mean = np.mean(q_vector)
    best_val = 0
    best_sent = ""
    print("----------", question['text'],"----------")
    print("Possible sentences")
    print(sentences)
    for sent in sentences:
        # A list of all the word tokens in the sentence
        bosw = norm_text(sent)
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(boqw & bosw)
        sent_vector = get_word_embedding_features(sent)        
        print(sent_vector)
        s_mean = np.mean(sent_vector)
#        print("---Cosine similarity---")
#        similarity = cos_sim(q_vector,sent_vector)
        m = np.asmatrix(q_vector)
        n = np.asmatrix(sent_vector)
        similarity = cosine_similarity(m,n)
        
        if similarity > best_val:
            best_val = similarity
            best_sent = sent
        print(similarity,"--->", sent)
#        print(similarity)        
        answers.append((overlap, sent, bosw))
    
    
    print("***WORD EMBED BEST SENTENCE***")
    print(best_val,": ",best_sent)
    print("\n")
    if len(best_sent) == 0:
        answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
        default_answer = (answers[0])[1]
        max_overlap = (answers[0])[0]
        best_answer = ''

        for answer in answers:
            if answer[0] == max_overlap and qword in answer[2]:
                if best_answer == '':
                    best_answer = answer[1]
                else:
                    best_answer = best_answer + ' ' + answer[1]

        if best_answer == '':
            best_answer = default_answer

    else:
        best_answer = best_sent
    # print(question['qid'])
    # print(best_answer)
    # print('\n')  
        question = question["text"]  

    """    
    if "where" in question.lower():
        candidate_sent = get_sentences(best_answer)
        chunker = nltk.RegexpParser(GRAMMAR)
        locations = find_candidates(candidate_sent, chunker)
        for loc in locations:
            best_answer = " ".join([token[0] for token in loc.leaves()])
    
    if "why" in question.lower():
        candidate_sent = get_sentences(best_answer)
        for sent in candidate_sent:
            for index,pair in enumerate(sent):
                if pair[0] == "because":
                    sent_split = sent[index:]
                    best_answer = ' '.join([word_pair[0] for word_pair in sent_split])
    
    if 'who' in question.lower():
        # print(best_answer)
        try:
            sgraph = story[s_type][sentences.index(best_answer)]
            sword  = find_main(sgraph)['lemma']
            node   = find_node(qword, sgraph)

            if node != None:
                best_answer = best_answer[:best_answer.index(node['word'])]
            else:
                pass
        except:
            pass
    
    """
    return best_answer

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None

def find_node(word, graph):
    for node in graph.nodes.values():
        if node['word'] != None:
            if lmtzr.lemmatize(node["word"], get_wordnet_pos(node["tag"])) == word:
                return node
    return None

def pp_filter(subtree):
    return subtree.label() == "PP"

def is_location(prep):
    return prep[0] in LOC_PP


def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations

    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)

    return locations



def find_candidates(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = find_locations(tree)
        candidates.extend(locations)

    return candidates

def sent_test():
    driver = QABase()
    q = driver.get_question("fables-01-2")
    story = driver.get_story(q["sid"])

    qgraph = q["dep"]
    qpar = q['par']
    qmain = find_main(qgraph)
    qword = qmain["word"]
    # qset = set()
    # for node in qgraph.nodes.values():
    #     if node['word'] not in stopwords or node['word'] == qword:
    #         if node['address'] != 0 and node['address'] != 1 and node['rel'] != 'punct':
    #             qset.add(node['lemma'])
            

    sgraph = story["sch_dep"]
    spar = story["story_par"]
    print(spar)
    given_sent =  "A Crow was sitting on a branch of a tree with a piece of cheese in her beak when a Fox observed her and set his wits to work to discover some way of getting the cheese."
    candidate_sent = get_sentences(given_sent)
    print(candidate_sent)
    print(qmain)
    
    chunker = nltk.RegexpParser(GRAMMAR)
    locations = find_candidates(candidate_sent, chunker)
    #print("location candidates")
    #print(locations)        
     # Print them out

    for loc in locations:
        print(loc)
        print(" ".join([token[0] for token in loc.leaves()]))

    """
    for sent in candidate_sent:
        tree = chunker.parse(sent)
        print(tree)
    # print(sgraph[0])
    """

#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evaluate=False):
    QA = QAEngine(evaluate=evaluate)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    #sent_test()
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
