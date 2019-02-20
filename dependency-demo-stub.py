#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

from qa_engine.base import QABase
    
def find_main(graph):
    print("find_main")
    print(graph)
    for node in graph.nodes.values():
        print(node)
        if node['rel'] == 'root':
            return node
    return None
    
def find_node(word, graph):
    print("Find node of -- ", word)
    for node in graph.nodes.values():
        if node["word"] == word:
            print("Found {} at {}".format(word, node))
            return node
    return None
    
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


def find_answer(qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    print("---qword---")
    print(qword)
    snode = find_node(qword, sgraph)
    print("---snode---")
    print(snode)

    for node in sgraph.nodes.values():
        #print("node[head]=", node["head"])
        if node.get('head', None) == snode["address"]:
            print("word, relation")
            print(node["word"], node["rel"])

            if node['rel'] == "nmod":
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                
                return " ".join(dep["word"] for dep in deps)


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("fables-01-1")
    story = driver.get_story(q["sid"])

    print("Question: ", q["text"])
    print("Story: ", story['sch'])

    # get the dependency graph of the first question
    qgraph = q["dep"]
    #print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo

    #sch_dep is dependency parses of Scherherrazade
    sgraph = story["sch_dep"][1]
    print("sgraph")
    print(sgraph)

    
    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        word = node["word"]
        tag = node["tag"]
#        print(word, tag)
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()

    answer = find_answer(qgraph, sgraph)
    print("Answer:", answer)

