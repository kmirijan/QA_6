import re, sys, nltk, operator
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from qa_engine.modified_score_answers import main as mod_score_answers
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer

stopwords = set(nltk.corpus.stopwords.words("english"))
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()

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

    start_word = quest_words[0][0]
    del quest_words[0]
    del quest_words[-1]

    root_question_words = set()
    
    # Adds the root word to troot_question_words
    qgraph = question["dep"]
    qword = find_main(qgraph)['lemma']
    if(qword != start_word.lower()):
        qverbs = wordnet._morphy(qword, wordnet.VERB)
        root_question_words.add(qword)
        for qverb in qverbs:
            root_question_words.add(qverb)
        
    for word_pair in quest_words:
        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag  = get_wordnet_pos(word_pair[1])
            
            #Added in morphy function and added stems to root_question_words
            stems = wordnet._morphy(word, tag)
            if not stems:
                stems.append(word)
            [root_question_words.add(stem) for stem in stems]
            if word_pair[1] == 'NNP':
                root_question_words.add(word)

    return root_question_words, qword

def norm_text(sent):
    sent_words = nltk.word_tokenize(sent)
    sent_words = nltk.pos_tag(sent_words)

    root_sent_words = set()

    for word_pair in sent_words:
        # If the word is a Verb, then add the word to root_sent_words
        if (word_pair[1].startswith('V')):
            words = wordnet._morphy(word_pair[0].lower(), wordnet.VERB)
            [root_sent_words.add(word) for word in wordnet._morphy(word_pair[0].lower(), wordnet.VERB)]

        if word_pair[0] not in stopwords:
            word = word_pair[0].lower()
            tag  = get_wordnet_pos(word_pair[1])
            
            #Added in morphy function and added stems to root_question_words
            stems = wordnet._morphy(word, tag)
            if not stems:
                stems.append(word)
            [root_sent_words.add(stem) for stem in stems]
            if word_pair[1] == 'NNP':
                root_sent_words.add(word)

    return root_sent_words

def get_question_words(qgraph, qword):

    qset = set()
    for node in qgraph.nodes.values():
        if node['word'] not in stopwords or node['word'] == qword:
            if node['address'] != 0 and node['address'] != 1 and node['rel'] != 'punct':
                qset.add(node['lemma'])
    return qset

def select_sentence(question, story, text, s_type):
    answers = []

    boqw, qword = norm_question(question)
    qgraph = question["dep"]
    qsubj = find_nsubj(qgraph)
    if qsubj is not None:
        qsubj = qsubj["lemma"]
    # print(question['text'])
    # print(boqw)
    # print(qword)

    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        # A list of all the word tokens in the sentence
        bosw = norm_text(sent)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(boqw & bosw)
        
        answers.append((overlap, sent, bosw))
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    default_answer = (answers[0])[1]
    max_overlap = (answers[0])[0]
    best_answer = ''

    for answer in answers:
        if answer[0] == max_overlap:
            ssubj = find_nsubj(story[s_type][sentences.index(answer[1])])
            if(ssubj is not None):
                ssubj = ssubj["lemma"]
            if qword in answer[2] and qsubj == ssubj:
                best_answer = set_best_answer(best_answer, answer)
            elif qword in answer[2]:
                best_answer = set_best_answer(best_answer, answer)
            elif qsubj == ssubj:
                best_answer = set_best_answer(best_answer, answer)

    if best_answer == '':
        best_answer = default_answer

    return best_answer

def parse_yes_no(question, best_answer):
    best_answer = 'yes'
    return best_answer

def parse_chunk_where(question, best_answer):
    candidate_sent = get_sentences(best_answer)
    chunker = nltk.RegexpParser(GRAMMAR)
    locations = find_candidates(candidate_sent, chunker)
    for loc in locations:
        best_answer = " ".join([token[0] for token in loc.leaves()])

    return best_answer

def get_words(text):
    words_list = []
    for sent in nltk.sent_tokenize(text):
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sent)):
            if word not in string.punctuation:
            #if word is not re.search(r'\w', word):
                words_list.append(word.lower())

    return words_list

def parse_where(question, story, sentences, given_sent, s_type):
    qgraph = question["dep"]
    q_main_node = find_main(qgraph)
    q_main_word = wordnet._morphy(q_main_node["lemma"], get_wordnet_pos(q_main_node["tag"]))
    #print("Q_MAIN_NODE:", q_main_node)

    cand_sents = nltk.sent_tokenize(given_sent)
    for cand_sent in cand_sents:
        ggraph = story[s_type][sentences.index(cand_sent)]
        g_main_node = find_main(ggraph)
        g_main_word = wordnet._morphy(g_main_node["lemma"], get_wordnet_pos(g_main_node["tag"]))
        nmod_deps = get_nmod_dep(g_main_node, ggraph)

        best_answer = ""
        chunk_ans = ""
        dep_ans =""
        if nmod_deps and g_main_word == q_main_word:
            for dep in nmod_deps:
                result = []
                result.append((dep["address"], dep["word"]))
                all_deps = get_nmod_dependents(dep, ggraph)
                for item in all_deps:
                    result.append((item["address"], item['word']))
                result = sorted(result)
                answer = ' '.join([word_pair[1] for word_pair in result])
                dep_ans = dep_ans + " " + answer
        else:
            answer = parse_chunk_where(question['text'], given_sent)
            chunk_ans = chunk_ans + " " + answer

        if dep_ans != "":
            return dep_ans
        elif chunk_ans != "":
            return chunk_ans
        else:
            return given_sent


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

    text = ''
    
    if question['type'] == "Story" or question['type'] == "Story|Sch":
        text = story['text']
        s_type = 'story_dep'
    elif question['type'] == "Sch" or question['type'] == "Sch|Story":
        text = story['sch']
        s_type = 'sch_dep'
    else:
        text = story['text']
        s_type = 'story_dep'

    sentences = nltk.sent_tokenize(text)

    best_answer = select_sentence(question, story,text, s_type)

    question_words = nltk.word_tokenize(question["text"])

    if question_words[0].lower() == "did" or question_words[0].lower() == "had":
        best_answer = parse_yes_no(question, best_answer)

    if question_words[0].lower() == "where":
        best_answer = parse_where(question, story, sentences, best_answer, s_type)
    
    if question_words[0] == "Why":
        #get a tokenized list of words from the question
        words = get_words(question["text"])
        #get last word of question sentence
        last_word = words[-1]
        candidate_sent = get_sentences(best_answer)
        best_answer1 = ""
        best_answer2 = ""
        for sent in candidate_sent:
            for index,pair in enumerate(sent):
                #if there are two sentences returned, typically always want 
                #the sentence returned after "because"
                if pair[0] == "because":
                    sent_split = sent[index:]
                    best_answer1 = ' '.join([word_pair[0] for word_pair in sent_split])
                    break
                #get words after the last word in the question
                else:
                    if stemmer.stem((pair[0]).lower()) == stemmer.stem(last_word):
                        val = index+1
                        sent_split = sent[val:]
                        best_answer = ' '.join([word_pair[0] for word_pair in sent_split if word_pair[0] not in string.punctuation])
        if best_answer1 != "":
            best_answer = best_answer1
        elif best_answer2 != "":
            best_answer = best_answer2
        else:
            best_answer = best_answer

    
    if question_words[0].lower() == "who":
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

    if question_words[0].lower() == "what":

        qgraph       = question["dep"]
        q_relations  = [[node['address'], node['word'], node['rel']] for node in qgraph.nodes.values() if node['rel'] != None]
        q_relations  = sorted(q_relations, key=lambda tup: tup[0])

        try:
            s_text, s_type, sentences = get_text_type(question, story)
            boqw, qword = norm_question(question)
            sgraph      = story[s_type][sentences.index(best_answer)]
            sword       = find_main(sgraph)['lemma']
            node        = find_node(qword, sgraph)

            if node != None:
                best_answer = nltk.word_tokenize(best_answer[best_answer.index(node['word']):])
                best_answer = ' '.join(word for word in best_answer[1:] if word.isalpha())
            else:
                pass
        except:
            pass
    
    return best_answer

def get_text_type(question, story):
    text   = ''
    s_type = ''
    sentences = ''

    if question['type'] == "Sch":
        text   = story['sch']
        s_type = 'sch_dep'
    else:
        text   = story['text']
        s_type = 'story_dep'
    
    sentences = nltk.sent_tokenize(text)

    return text, s_type, sentences

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None

def find_nsubj(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'nsubj':
            return node
    return None

def find_nmod(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'nmod':
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

def set_best_answer(best_answer, answer):
    if best_answer == '':
        best_answer = answer[1]
    else:
        best_answer = best_answer + ' ' + answer[1]
    return best_answer

def find_candidates(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = find_locations(tree)
        candidates.extend(locations)

    return candidates

def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
    #print("get dependents")
    #print(results)
    return results

def get_nmod_dependents(node, graph):
    results = []
    for item in node["deps"]:
        if item != "advcl" and item != "mwe":
            address = node["deps"][item][0]
            dep = graph.nodes[address]
            results.append(dep)
            results = results + get_nmod_dependents(dep, graph)
    #print("get dependents")
    #print(results)
    return results

def get_nmod_dep(node, graph):
    results = []
    for item in node["deps"]:
        if item == "nmod":
            address = node["deps"][item][0]
            dep = graph.nodes[address]
            results.append(dep)
    #print("get dependents")
    #print(results)
    return results

def sent_test():
    driver = QABase()
    q = driver.get_question("mc500.train.23.17")
    story = driver.get_story(q["sid"])

    qgraph = q["dep"]
    # qpar = q['par']
    # qmain = find_main(qgraph)
    # qword = qmain["word"]
    # qset = set()
    # for node in qgraph.nodes.values():
    #     if node['word'] not in stopwords or node['word'] == qword:
    #         if node['address'] != 0 and node['address'] != 1 and node['rel'] != 'punct':
    #             qset.add(node['lemma'])
            
    given_sent =  'Andrew looked outside the window and saw the newspaper by the door.'
    
    answers = []
    boqw, qword = norm_question(q)
    qsubj = find_nsubj(qgraph)["lemma"]

    if q['type'] == "Story" or q['type'] == "Story|Sch":
        text = story['text']
        s_type = 'story_dep'
    elif q['type'] == "Sch" or q['type'] == "Sch|Story":
        text = story['sch']
        s_type = 'sch_dep'

    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        # A list of all the word tokens in the sentence
        bosw = norm_text(sent)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(boqw & bosw)
        
        answers.append((overlap, sent, bosw))
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    default_answer = (answers[0])[1]
    max_overlap = (answers[0])[0]
    best_answer = ''

    for answer in answers:
        if answer[0] == max_overlap:
            ssubj = find_nsubj(story[s_type][sentences.index(answer[1])])["lemma"]
            if qword in answer[2] and qsubj == ssubj:
                best_answer = set_best_answer(best_answer, answer)
            elif qword in answer[2]:
                best_answer = set_best_answer(best_answer, answer)
            elif qsubj == ssubj:
                best_answer = set_best_answer(best_answer, answer)
                
    if best_answer == '':
        best_answer = default_answer

    # bgraph = story[s_type][sentences.index(best_answer)]
    # ggraph = story[s_type][sentences.index(given_sent)]

    print(q["text"])
    print(qword)
    print(boqw)
    print(best_answer)
    print(norm_text(best_answer))
    print(len(boqw & norm_text(best_answer)))
    print(given_sent)
    print(norm_text(given_sent))
    print(len(boqw & norm_text(given_sent)))

    # print(qgraph)
    # print(bgraph)
    # print(ggraph)
    # print(text)



    """
    for sent in candidate_sent:
        tree = chunker.parse(sent)
        print(tree)
    # print(sgraph[0])
    """
def parse_test():
    driver = QABase()
    q = driver.get_question("blogs-03-5")
    story = driver.get_story(q["sid"])

    if q['type'] == "Story" or q['type'] == "Story|Sch":
        text = story['text']
        s_type = 'story_dep'
    elif q['type'] == "Sch" or q['type'] == "Sch|Story":
        text = story['sch']
        s_type = 'sch_dep'
    sentences = nltk.sent_tokenize(text)

    given_sent = 'Once, storm was occurring in Louisiana. Storm occurred.'

    # qgraph = q["dep"]
    # print("QUESTION:",q['text'])
    # q_main_node = find_main(qgraph)
    # q_main_word = wordnet._morphy(q_main_node["lemma"], get_wordnet_pos(q_main_node["tag"]))
    # #print("Q_MAIN_NODE:", q_main_node)

    # cand_sents = nltk.sent_tokenize(given_sent)
    # for cand_sent in cand_sents:
    #     ggraph = story[s_type][sentences.index(cand_sent)]
    #     g_main_node = find_main(ggraph)
    #     print("CAND_SENT",cand_sent)
    #     g_main_word = wordnet._morphy(g_main_node["lemma"], get_wordnet_pos(g_main_node["tag"]))
    #     nmod_deps = get_nmod_dep(g_main_node, ggraph)
    #     best_answer = ""
    #     chunk_ans = ""
    #     dep_ans =""
    #     print(g_main_word, q_main_word)
    #     if nmod_deps and g_main_word == q_main_word:
    #         for dep in nmod_deps:
    #             result = []
    #             result.append((dep["address"], dep["word"]))
    #             all_deps = get_nmod_dependents(dep, ggraph)
    #             for item in all_deps:
    #                 result.append((item["address"], item['word']))
    #             result = sorted(result)
    #             answer = ' '.join([word_pair[1] for word_pair in result])
    #             dep_ans = dep_ans + " " + answer
    #             print("DEP_BEST:",answer)
    #     else:
    #         answer = parse_chunk_where(q['text'], given_sent)
    #         chunk_ans = chunk_ans + " " + answer
    #         print("CHUNK_BEST:", answer)

    #     if dep_ans != "":
    #         print(dep_ans)
    #     elif chunk_ans != "":
    #         print(chunk_ans)
    #     else:
    #         print(given_sent)

    print(parse_where(q, story, sentences, given_sent, s_type))


    #print(parse_where(q['text'], given_sent))






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
    #parse_test()
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()
    #mod_score_answers(print_story=False)

if __name__ == "__main__":
    main()
