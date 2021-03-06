import re, sys, nltk, operator, csv, string, os, json
from nltk.corpus import wordnet
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from qa_engine.modified_score_answers import main as mod_score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from word2vec_extractor import Word2vecExtractor
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import numpy as np

DATA_DIR = "./wordnet"

stopwords = set(nltk.corpus.stopwords.words("english"))
lmtzr = WordNetLemmatizer()
stemmer = PorterStemmer()
glove_w2v_file = "data/glove-w2v.txt"
W2vecextractor = Word2vecExtractor(glove_w2v_file)


GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

GRAMMARVERB =   """
                N: {<PRP>|<NN.*>}
                V: {<V.*>}
                ADJ: {<JJ.*>}
                NP: {<DT>? <ADJ>* <N>+ <V.*>+ ((<ADJ> <CC> <V>)|<NN.*>|(<RP>? <PP>))*}
                PP: {<IN> <DT>? (<NP>|<NNP>)}
                VP: {<TO>? <V> (<NP>|<PP>)*}
                """

LOC_PP = set(["in", "from","on", "at", "along","under", "around","near","to","in front of"])

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

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

def get_wordnet_words(question):
    sent_words = get_words(question["text"], True)
    answer = ' '.join([word for word in sent_words])

    all_synset_ids = []
    wordnet_set = set()
    for k, v in verb_ids.items():
        stories = (v["stories"])
        stories = stories.replace("\'", '')
        stories = stories.replace(".vgl", '')
        stories = stories.replace(",", '')
        stories = nltk.word_tokenize(stories)
        for story in stories:
            if question['sid'] == story:
                all_synset_ids.append(k)

    for k, v in noun_ids.items():
        stories = (v["stories"])
        stories = stories.replace("\'", '')
        stories = stories.replace(".vgl", '')
        stories = stories.replace(",", '')
        stories = nltk.word_tokenize(stories)
        for story in stories:
            if question['sid'] == story:
                all_synset_ids.append(k)

    for synset_id in all_synset_ids:
        for synset in wn.synset(synset_id).hyponyms():
            all_synset_ids.append(synset.name())
        # for sysnet in wn.synset(synset_id).hypernyms():
        #     all_synset_ids.append(synset.name())

    for synset_id in all_synset_ids:
        synsets = wn.synset(synset_id)
        synset_lemmas_list = synsets.lemma_names()
        for lemma in synset_lemmas_list:
#            print(lemma)
            val = lemma.replace('_', " ")
            #print("val:", val)
            if val in answer:
                #print("answer:", answer)
                wordnet_set.add(synset_id.split(".", 1)[0])
                
    return wordnet_set

def norm_question(question):
    quest_words = nltk.word_tokenize(question['text'])
    quest_words = nltk.pos_tag(quest_words)

    start_word = quest_words[0][0]
    del quest_words[0]
    del quest_words[-1]

    root_question_words = get_wordnet_words(question)
    

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
            try:
                ssubj = find_nsubj(story[s_type][sentences.index(answer[1])])
            except IndexError:
                print("Index Error", question["qid"])
                ssubj = None

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

def baseline_word2vec(question, sentences, stopwords, W2vecextractor):
    q_feat = W2vecextractor.sent2vec(question)
    candidate_answers = []

    for sent in sentences:
       a_feat = W2vecextractor.sent2vec(sent)
       dist = cosine_similarity(q_feat, a_feat)[0]
       candidate_answers.append((dist, sent))
       #print("distance: "+str(dist)+"\t sent: "+sent)

    answers = sorted(candidate_answers, key=operator.itemgetter(0), reverse=True)

    print(answers)

    best_answer = (answers[0])[1]
    return best_answer


def baseline_word2vec_verb(question, sentences, stopwords, W2vecextractor, q_verb, sgraphs):
    q_feat = W2vecextractor.word2v(q_verb)
    candidate_answers = []
    print("ROOT of question: "+str(q_verb))

    for i in range(0, len(sentences)):
        sent = sentences[i]
        s_verb = find_main(sgraphs[i])['word']
        print("ROOT of sentence: "+str(s_verb))
        a_feat = W2vecextractor.word2v(s_verb)

        dist = cosine_similarity([q_feat], [a_feat])
        candidate_answers.append((dist[0], sent))


    answers = sorted(candidate_answers, key=operator.itemgetter(0), reverse=True)

    print(answers)

    best_answer = (answers[0])[1]
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

def get_words(text, do_lemmas=False):
    words_list = []
    lem_word_list = []
    for sent in nltk.sent_tokenize(text):
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sent)):
            if word not in string.punctuation:
            #if word is not re.search(r'\w', word):
                lem_word_list.append(lmtzr.lemmatize(word, get_wordnet_pos(pos)))
                words_list.append(word.lower())

    if do_lemmas == False:
        return words_list
    elif do_lemmas == True:
        return lem_word_list

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

def parse_who(question, story, sentences, given_sent, s_type, c_type):
    qgraph = question["dep"]
    q_main_node = find_main(qgraph)
    q_main_word = wordnet._morphy(q_main_node["lemma"], get_wordnet_pos(q_main_node["tag"]))
    exclude = set(string.punctuation)

    cand_sents = nltk.sent_tokenize(given_sent)
    cand_subtrees = []
    q_main_answer = ""
    q_subj_answer = ""

    for cand_sent in cand_sents:
        gcon = story[c_type][sentences.index(cand_sent)]
        for subtree in gcon.subtrees():
            if subtree.label() == "NP":
                tree_pos = subtree.pos()
                for tree_pair in tree_pos:
                    if lmtzr.lemmatize(tree_pair[0], get_wordnet_pos(tree_pair[1])) in q_main_word:
                        cand_subtrees.append(subtree)

        for tree in cand_subtrees:
            for word_pair in tree.pos():
                if lmtzr.lemmatize(word_pair[0], get_wordnet_pos(word_pair[1])) not in q_main_word and word_pair[0] not in exclude:
                    q_main_answer = q_main_answer + " " + word_pair[0]

    for cand_sent in cand_sents:
        gcon = story[c_type][sentences.index(cand_sent)]
        ggraph = story[s_type][sentences.index(cand_sent)]
        all_nsubj = find_all_nsubj(ggraph)
        for subtree in gcon.subtrees():
            if subtree.label() == "NP" and subtree.height() < 4:
                tree_pos = subtree.pos()
                for nsubj in all_nsubj:
                    if nsubj in tree_pos and len(tree_pos) > 1:
                        for word_pair in tree_pos:
                            q_subj_answer = q_subj_answer + " " + word_pair[0]


    if q_main_answer != "":
        return q_main_answer
    elif q_subj_answer != "":
        return q_subj_answer
    else:
        return given_sent


def parse_why(question, story, sentences, best_answer, s_type):
    
    #get a tokenized list of words from the question
    words = get_words(question["text"])
    #get last word of question sentence
    last_word = words[-1]
    candidate_sent = get_sentences(best_answer)
    best_answer1 = ""
    best_answer2 = ""
    print("Question:", question["text"])
    print(candidate_sent)
    for sent in candidate_sent:
        for index,pair in enumerate(sent):
            #if there are two sentences returned, typically always want 
            #the sentence returned after "because"
            if pair[0] == "because":
                print("found because")
                sent_split = sent[index:]
                best_answer1 = ' '.join([word_pair[0] for word_pair in sent_split])
                print("Best_answer1:", best_answer1)
                best_answer = best_answer1
                return best_answer
                #get words after the last word in the question
            else:
                if stemmer.stem((pair[0]).lower()) == stemmer.stem(last_word):
                    val = index+1
                    sent_split = sent[val:]
                    best_answer2 = ' '.join([word_pair[0] for word_pair in sent_split if word_pair[0] not in string.punctuation])
    if best_answer1 != "":
        best_answer = best_answer1
    elif best_answer2 != "":
        best_answer = best_answer2
    else:
        best_answer = best_answer

    print("Best answer")
    print(best_answer)
    return best_answer 


def parse_what_verb(question, story, sentences, best_answer, s_type):
    print("Question: ", question["text"])
    boqw, qword = norm_question(question)
    print("qword:", qword)
    print("Possible sentences")

    candidate_sent = get_sentences(best_answer)
    print(candidate_sent) 
    chunker = nltk.RegexpParser(GRAMMARVERB)
    locations = find_candidates_verb(candidate_sent, chunker, qword)
    for loc in locations:
        print(loc)
        print(" ".join([token[0] for token in loc.leaves()]))
    
    
    print("\n")
    return best_answer

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
    
    if question['type'] == "Story" or question['type'] == "Story | Sch" or question['type'] == "Story|Sch":
        text = story['text']
        s_type = 'story_dep'
        c_type = 'story_par'
    elif question['type'] == "Sch" or question['type'] == "Sch | Story" or question['type'] == "Sch|Story":
        text = story['sch']
        s_type = 'sch_dep'
        c_type = 'sch_par'

    sentences = nltk.sent_tokenize(text)
    best_answer = select_sentence(question, story,text, s_type)
    default_answer = best_answer

    question_words = nltk.word_tokenize(question["text"])

    if question_words[0].lower() == "did" or question_words[0].lower() == "had":
        best_answer = parse_yes_no(question, best_answer)

    if question_words[0].lower() == "where":
        best_answer = parse_where(question, story, sentences, best_answer, s_type)
    
    if question_words[0].lower() == "why":
        best_answer = parse_why(question, story, sentences,  best_answer, s_type)
        
    if question_words[0].lower() == "who":
        best_answer = parse_who(question, story, sentences, best_answer, s_type, c_type)
        # try:
        #     sgraph = story[s_type][sentences.index(best_answer)]
        #     sword  = find_main(sgraph)['lemma']
        #     node   = find_node(qword, sgraph)

        #     if node != None:
        #         best_answer = best_answer[:best_answer.index(node['word'])]
        #     else:
        #         pass
        # except:
        #     pass
    if question_words[0].lower() == "what":
        question_phrase = question_words[0] + " " + question_words[1]
        strdid = "What did"
        strwas = "What was"
        if question_phrase not in strdid or question_phrase not in strwas:
            print("Did not found did or was")     
            best_answer = parse_what_verb(question, story, sentences, best_answer, s_type)
        else:
            return best_answer
            
        """
        qgraph       = question["dep"]
        q_relations  = [[node['address'], node['word'], node['rel']] for node in qgraph.nodes.values() if node['rel'] != None]
        q_relations  = sorted(q_relations, key=lambda tup: tup[0])

        # try:
        #     s_text, s_type, sentences = get_text_type(question, story)
        #     boqw, qword = norm_question(question)
        #     sgraph      = story[s_type][sentences.index(best_answer)]
        #     sword       = find_main(sgraph)['lemma']
        #     node        = find_node(qword, sgraph)

            if node != None:
                best_answer = nltk.word_tokenize(best_answer[best_answer.index(node['word']):])
                best_answer = ' '.join(word for word in best_answer[1:] if word.isalpha())
            else:
                pass
        except:
            pass
        """

    if default_answer == best_answer:
        for word in question_words:
            best_answer = best_answer.replace(word, "")
    
    return best_answer

def parse_what(question, story, sentences, given_sent, s_type, c_type):

    qgraph = question["dep"]
    q_main_node = find_main(qgraph)
    q_subj_node = find_nsubj(qgraph)
    q_main_word = wordnet._morphy(q_main_node["lemma"], get_wordnet_pos(q_main_node["tag"]))
    cand_sents = nltk.sent_tokenize(given_sent)
    cand_subtrees = []
    answer = ""

    for cand_sent in cand_sents:
        if cand_sent == "The oven malfunctioned... it was supposed to be baking at 350 degrees but somehow broiler came on and got stuck?":
            cand_sent = cand_sent + "?..."
        if cand_sent == "?...":
            continue
        lemma_words = get_words(cand_sent, True)
        ggraph = story[s_type][sentences.index(cand_sent)]
        gcon = story[c_type][sentences.index(cand_sent)]

        if q_main_node is not None and q_subj_node is not None and q_main_node['lemma'] in lemma_words and q_subj_node['lemma'] in lemma_words:
            indx = lemma_words.index(q_main_node['lemma'])
            just_words = get_words(cand_sent)
            answer_list = just_words[indx+1:]
            for word in answer_list:
                answer = answer + " " + word

        elif q_main_node is not None and q_main_node['lemma'] in lemma_words:
            indx = lemma_words.index(q_main_node['lemma'])
            just_words = get_words(cand_sent)
            answer_list = just_words[:indx]
            for word in answer_list:
                answer = answer + " " + word
        #     for postn in gcon.treepositions():
        #         if type(gcon[postn]) is not str:
        #             if gcon[postn].label() == "VP" and gcon[postn].height() > 3:
        #                 sub_pos = gcon[postn].pos()
        #                 sub_lemma = []
        #                 for pair in sub_pos:
        #                     sub_lemma.append(lmtzr.lemmatize(pair[0], get_wordnet_pos(pair[1])))
        #                 if q_main_node["lemma"] in sub_lemma and (postn[-1] - 1) >= 0:
        #                     brother_postn = postn[:-1] + ((postn[-1] - 1),)
        #                     answer_list = gcon[brother_postn].leaves()
        #                     for word in answer_list:
        #                         answer = answer + " " + word


    if answer != "":
        return answer
    else:
        return given_sent

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

def find_all_nsubj(graph):
    nodes = []
    for node in graph.nodes.values():
        if node['rel'] == 'nsubj':
            nodes.append((node['lemma'], node['tag']))
    return nodes

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

def verb_filter(subtree):
    return subtree.label() == "VP"

def is_location(prep):
    return prep[0] in LOC_PP 

def is_location_verb(prep, qword):
    print("is location")
    print("prep")
    print(prep)
    verbset = [qword]
    return prep[0] in verbset

def noun_filter(subtree):
    return subtree.label() == "NP"

def find_locations(tree):
    print("find locations")
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
        print(subtree)
        if is_location(subtree[0]):
            locations.append(subtree)

    return locations


def find_locations_verb(tree, qword):
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
    #for subtree in tree.subtrees(filter=pp_filter):
    print("In find_locations")
    for subtree in tree.subtrees(filter=noun_filter): 
        print("subtree")
        print(subtree)
        for sub in subtree:
            locations.append(sub)

#        locations.append(subtree)
#        if is_location(subtree[0], qword):
#            locations.append(subtree)
        #if qword in subtree[0]:
        #    print("correct subtree")
        #    print(subtree)
        #    print("right vphrase")
        #    print(subtree[0])
    #  if is_location(subtree[0]):
     #       locations.append(subtree)

    print("locations")
    print(locations)
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

def find_candidates_verb(sentences, chunker, qword):
    print("find candidates")
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        print(tree)

        locations = find_locations_verb(tree, qword)
        #candidates.extend(locations)

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

def load_coref_json(full_data = False):
    """

    Returns the coreferences of every story as a dictionary with the key 
    being the story id, listed in COREF STORIES. The value of this dict is 
    the coreferences of the story contained in another dictionary, where the key 
    is the entity id, and the values are all mentions of that entity. It is probably
    best to  print out the dictionaries and see what they look like. 
    This is an intermediate step to use corefs in our answer selection.

    Set full_data to True to receive full results of Stanford parser.


    """

    COREF_DICT = {}
    
    COREF_STORIES = ['blogs-01.json'      , 'blogs-02.json'      , 'blogs-03.json' , 
                     'blogs-04.json'      , 'blogs-05.json'      , 'blogs-06.json' , 
                     'fables-01.json'     , 'fables-02.json'     , 'fables-03.json', 
                     'fables-04.json'     , 'fables-05.json'     , 'fables-06.json', 
                     'mc500.train.0.json' , 'mc500.train.18.json', 'mc500.train.23.json', 
                     'mc500.train.25.json', 'mc500.train.111.json'
    ]

    for temp_story in COREF_STORIES:
        # print(temp_story[:-5])
        with open(os.path.join('coref_json', temp_story)) as raw_json:
            data        = raw_json.read()
            story_key   = temp_story[:-5]
            result_dict = json.loads(data)

            if full_data == True:
                COREF_DICT[story_key] = result_dict
            else:
                COREF_DICT[story_key] = result_dict['corefs']

    return COREF_DICT


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
        bows.append('strike')        
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
    q = driver.get_question("blogs-06-6")
    story = driver.get_story(q["sid"])

    if q['type'] == "Story" or q['type'] == "Story | Sch":
        text = story['text']
        s_type = 'story_dep'
        c_type = 'story_par'
    elif q['type'] == "Sch" or q['type'] == "Sch | Story":
        text = story['sch']
        s_type = 'sch_dep'
        c_type = 'sch_par'

    sentences = nltk.sent_tokenize(text)

    given_sent = 'The oven malfunctioned... it was supposed to be baking at 350 degrees but somehow broiler came on and got stuck??...'

    qgraph = q["dep"]
    print("QUESTION:",q['text'])
    q_main_node = find_main(qgraph)
    q_subj_node = find_nsubj(qgraph)
    q_main_word = wordnet._morphy(q_main_node["lemma"], get_wordnet_pos(q_main_node["tag"]))
    print("Q_MAIN_WORD:", q_main_word)
    exclude = set(string.punctuation)
    cand_sents = nltk.sent_tokenize(given_sent)
    cand_subtrees = []
    answer = ""

    for cand_sent in cand_sents:
        if cand_sent == "The oven malfunctioned... it was supposed to be baking at 350 degrees but somehow broiler came on and got stuck?":
            cand_sent = cand_sent + "?..."
        if cand_sent == "?...":
            continue
        lemma_words = get_words(cand_sent, True)
        ggraph = story[s_type][sentences.index(cand_sent)]
        gcon = story[c_type][sentences.index(cand_sent)]
        if q_main_node is not None and q_subj_node is not None and q_main_node['lemma'] in lemma_words and q_subj_node['lemma'] in lemma_words:
            print("ROOT AND NSUBJ", cand_sent)
            indx = lemma_words.index(q_main_node['lemma'])
            print(indx)
            just_words = get_words(cand_sent)
            answer_list = just_words[indx+1:]
            for word in answer_list:
                answer = answer + " " + word
            print(answer)

        elif q_main_node is not None and q_main_node['lemma'] in lemma_words:
            print("ROOT", cand_sent)
            print(gcon)
            for postn in gcon.treepositions():
                if type(gcon[postn]) is not str:
                    if gcon[postn].label() == "VP" and gcon[postn].height() > 3:
                        sub_pos = gcon[postn].pos()
                        sub_lemma = []
                        for pair in sub_pos:
                            sub_lemma.append(lmtzr.lemmatize(pair[0], get_wordnet_pos(pair[1])))
                        if q_main_node["lemma"] in sub_lemma and (postn[-1] - 1) >= 0:
                            brother_postn = postn[:-1] + ((postn[-1] - 1),)
                            answer_list = gcon[brother_postn].leaves()
                            for word in answer_list:
                                answer = answer + " " + word
                                print(answer)

    # for cand_sent in cand_sents:
    #     gcon = story[c_type][sentences.index(cand_sent)]
    #     for subtree in gcon.subtrees():
    #         if subtree.label() == "NP":
    #             tree_pos = subtree.pos()
    #             for tree_pair in tree_pos:
    #                 if lmtzr.lemmatize(tree_pair[0], get_wordnet_pos(tree_pair[1])) in q_main_word:
    #                     cand_subtrees.append(subtree)

    #     for tree in cand_subtrees:
    #         print(tree)
    #         for word_pair in tree.pos():
    #             if lmtzr.lemmatize(word_pair[0], get_wordnet_pos(word_pair[1])) not in q_main_word and word_pair[0] not in exclude:
    #                 q_main_answer = answer + " " + word_pair[0]

    # for cand_sent in cand_sents:
    #     gcon = story[c_type][sentences.index(cand_sent)]
    #     ggraph = story[s_type][sentences.index(cand_sent)]
    #     all_nsubj = find_all_nsubj(ggraph)
    #     for subtree in gcon.subtrees():
    #         if subtree.label() == "NP" and subtree.height() < 4:
    #             tree_pos = subtree.pos()
    #             for nsubj in all_nsubj:
    #                 if nsubj in tree_pos and len(tree_pos) > 1:
    #                     for word_pair in tree_pos:
    #                         q_subj_answer = q_subj_answer + " " + word_pair[0]

        # if nmod_deps and g_main_word == q_main_word:
        #     for dep in nmod_deps:
        #         result = []
        #         result.append((dep["address"], dep["word"]))
        #         all_deps = get_nmod_dependents(dep, ggraph)
        #         for item in all_deps:
        #             result.append((item["address"], item['word']))
        #         result = sorted(result)
        #         answer = ' '.join([word_pair[1] for word_pair in result])
        #         dep_ans = dep_ans + " " + answer
        #         print("DEP_BEST:",answer)

        # if dep_ans != "":
        #     print(dep_ans)
        # else:
        #     print(given_sent)

    #print(parse_where(q, story, sentences, given_sent, s_type))


def wnet_test():
    driver = QABase()
    q = driver.get_question("fables-06-16")
    story = driver.get_story(q["sid"])    

    sent_words = get_words(q["text"], True)
    answer = ' '.join([word for word in sent_words]) 
    
    all_synset_ids = []

    for k, v in verb_ids.items():
        stories = (v["stories"])
        stories = stories.replace("\'", '')
        stories = stories.replace(".vgl", '')
        stories = stories.replace(",", '')
        stories = nltk.word_tokenize(stories)
        for story in stories:
            if q['sid'] == story:
                all_synset_ids.append(k)

    for synset_id in all_synset_ids:
        for synset in wn.synset(synset_id).hyponyms():
            all_synset_ids.append(synset.name())
        # for sysnet in wn.synset(synset_id).hypernyms():
        #     all_synset_ids.append(synset.name())

    answer_set = set()

    for synset_id in all_synset_ids:
        synsets = wn.synset(synset_id)
        synset_lemmas_list = synsets.lemma_names()
        for lemma in synset_lemmas_list:
            val = lemma.replace('_', " ")
            if val in answer:
                answer_set.add(synset_id.split(".", 1)[0]) 

    print(answer_set)





#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evaluate=False):
    load_coref_json()
    QA = QAEngine(evaluate=evaluate)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    #parse_test()
    run_qa(evaluate=False)
    #wnet_test()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()
    mod_score_answers(print_story=False)

if __name__ == "__main__":
    main()
