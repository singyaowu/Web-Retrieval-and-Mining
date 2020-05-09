import argparse
import numpy as np
import xml.etree.ElementTree as ET
import csv
from copy import deepcopy
b_TF = 0.287
k_TF = 1.2

def docLenNormalize(vocIDList, countTermDoc, docLen, avdl):
    for term in vocIDList:
        for doc_id in countTermDoc[term]:
            count = countTermDoc[term][doc_id]
            raw_TF = (k_TF+1) * count / (count+k_TF)
            normalizer = 1 - b_TF + b_TF * docLen[doc_id]/avdl
            #TF = (k_TF+1) * count / (count+k_TF * normalizer)
            countTermDoc[term][doc_id] = raw_TF / normalizer
    return countTermDoc

def parse_query(query_path):
    tree = ET.parse(query_path)
    root = tree.getroot()
    queries_id = [(node.text).strip('\n')[-3:] for node in root.iter(tag='number')]
    query_concept = [(node.text).strip('。\n').split('、') for node in root.iter(tag='concepts')]
    return queries_id, query_concept

def RocchioFB(query, rankScore, countTermDoc,countDocTerm, TFTermDoc, numDoc, avdl):
    alpha, beta, gamma = 1, 0.3, 0.3
    sizeR, sizeNR = 15, 15
    
    doc_r = rankScore[:sizeR]
    doc_nr = rankScore[100:(100+sizeNR)]
    new_query = {}
    for doc_id, _ in doc_r:
        for term in countDocTerm[doc_id]:
            if term in new_query: new_query[term] += beta * TFTermDoc[term][doc_id] / sizeR
            else: 
                new_query[term] = beta * TFTermDoc[term][doc_id] / sizeR     
    for doc_id, _ in doc_nr:
        for term in countDocTerm[doc_id]:
            if term in new_query: new_query[term] -= gamma * TFTermDoc[term][doc_id] / sizeNR
            else: 
                new_query[term] = -gamma * TFTermDoc[term][doc_id] / sizeNR
    for term in set(list(query.keys())+list(new_query.keys())):
        if term in query:
            if term in new_query: new_query[term] += query[term] * alpha
            else: new_query[term] = query[term] * alpha
    scores = {}
    for term in new_query:
        IDF = np.log( (numDoc+1) / float(len(TFTermDoc[term])) )
        for doc in TFTermDoc[term]:
            TF_query = (k_TF+1) * new_query[term] / (new_query[term]+k_TF)
            score = (IDF * TF_query) * (TFTermDoc[term][doc])
            if doc not in scores: scores[doc] = score
            else: scores[doc] += score
    new_rankScore = sorted(scores.items(), key = lambda x:x[1], reverse=True)
    
    assert new_rankScore is not None
    #print(new_rankScore)
    return new_rankScore

def rank2Str(query_id, rankScore, fileListDocName):
    ret = query_id + ','
    count = 0
    for doc_id, _ in rankScore:
        filename = fileListDocName[doc_id].split('/')[-1].lower()
        ret += filename + ' '
        count += 1
        if count >= 100: break
    return ret[:-1]

def savecsv(output, outputPath):
    with open(outputPath, 'w') as fp:
        fp.write('query_id,retrieved_docs\n')
        for line in output: fp.write(line+'\n')
    return

def parse_invFile(invFile_path):
    with open(invFile_path, 'r', encoding='utf-8') as fp:
        invFile = [ [int(num) for num in line.strip('\n').split(' ')] for line in fp.readlines()]
    countTermDoc = {}
    countDocTerm = {}
    vocIDList = []
    docLen = np.zeros(len(fileListDocName)+1, dtype=float)
    i_inv = 0
    while i_inv < len(invFile):
        assert len(invFile[i_inv]) == 3
        voc_id1, voc_id2, numline = invFile[i_inv]
        if voc_id2 == -1: vocIDList.append(vocab[voc_id1])
        else: vocIDList.append(vocab[voc_id1] + vocab[voc_id2])
        
        countTermDoc[vocIDList[-1]] = {}
        i_inv += 1
        for j in range(numline):
            assert len(invFile[i_inv+j]) == 2
            doc_id, count = invFile[i_inv+j]
            countTermDoc[vocIDList[-1]][doc_id] = float(count)
            #if doc_id not in countDocTerm:
            #    countDocTerm[doc_id] = {}
            #countDocTerm[doc_id][vocIDList[-1]] = float(count)
            docLen[doc_id] += float(count)
        i_inv += numline
    for term in countTermDoc:
        for doc in countTermDoc[term]:
            if doc not in countDocTerm:
                countDocTerm[doc] = {}
            countDocTerm[doc][term] = countTermDoc[term][doc]
    return countTermDoc, countDocTerm, vocIDList, docLen

def toHalfChar(term):
    ret = ""
    for c in term:
        num = ord(c)
        if num == 12288: num = 32 
        elif (num >= 65281 and num <= 65374): num -= 65248
        ret += chr(num)
    return ret

def query_expand(concepts):
    query = {}
    for term in concepts:
        for i in range(len(term)):
            if ( term[i] >= '\u4e00' and term[i] <= '\u9fff'):# chinese
                unigram = term[i]
                if (unigram in vocIDList):
                    if (unigram not in query): query[unigram] = 1.0
                    else: query[unigram] += 1.0
                if i < len(term)-1:
                    bigram = term[i]+term[i+1]
                    if (bigram in vocIDList):
                        if (bigram not in query): query[bigram] = 1.0
                        else: query[bigram] += 1.0
            else:
                if term in vocIDList:
                    if (term not in query): query[term] = 1.0
                    else: query[term] += 1.0
                word = toHalfChar(term)
                if (word in vocIDList):
                    if (word not in query): query[word] = 1.0
                    else: query[word] += 1.0            
    return query

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='vsm based IR searching System')
    parser.add_argument('-r', dest='revelance', default=False, action='store_true')
    parser.add_argument('-b', dest='best_version', default=False, action='store_true')
    parser.add_argument('-i', dest='query_file', default='queries/query-train.xml')
    parser.add_argument('-o', dest='ranked_file', default='kaggle.csv')
    parser.add_argument('-m', dest='model_dir', default='model')
    parser.add_argument('-d', dest='NTCIR_dir', default='CIRB010')
    args = parser.parse_args()

    with open(args.model_dir + '/file-list', 'r', encoding='utf-8') as fp:
        fileListDocName = [line.strip('\n') for line in fp.readlines()]
    
    with open(args.model_dir + '/vocab.all', 'r', encoding='utf-8') as fp:
        vocab = [line.strip('\n') for line in fp.readlines()]

    numDoc = len(fileListDocName)
    
    print('====== Reading inverted-file ======')
    try:
        countTermDoc = np.load('countTermDoc.npy').item()
        countDocTerm = np.load('countDocTerm.npy').item()
        vocIDList = np.load('vocIDList.npy').tolist()
        docLen = np.load('docLen.npy')
        print('--- get saved inverted-file ---')
    except:
        print('--- Reading inverted-file ---')
        countTermDoc, countDocTerm, vocIDList,docLen = parse_invFile(args.model_dir+'/inverted-file')
        np.save('countTermDoc.npy', countTermDoc)
        np.save('countDocTerm.npy', countDocTerm)
        np.save('vocIDList.npy', vocIDList)
        np.save('docLen.npy', docLen)

    avdl = np.sum(docLen) / numDoc
    print('avdl: ', avdl)
    TFTermDoc = docLenNormalize(vocIDList, deepcopy(countTermDoc), docLen, avdl)
    
    print('====== Reading Queries ======')
    queries_id, query_concept = parse_query(args.query_file)
    
    print('====== Ranking Scores ======')
    results = []
    for i in range(len(queries_id)):
        query = query_expand(query_concept[i])
        scores = {}
        for term in query:
            IDF = np.log( (numDoc+1) / float(len(TFTermDoc[term])) )
            for doc in TFTermDoc[term]:
                TF_query = (k_TF+1) * query[term] / (query[term]+k_TF)
                score = (IDF * TF_query) * (TFTermDoc[term][doc])
                if doc not in scores: scores[doc] = score
                else: scores[doc] += score
        rankScore = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        if args.revelance:
            print('====== Use Rocchio Feedback ======')
            rankScore = RocchioFB(query, rankScore, countTermDoc,countDocTerm, TFTermDoc, numDoc, avdl)
        results.append(rank2Str(queries_id[i], rankScore, fileListDocName))
    
    print('====== Saving Output File ======')
    savecsv(results, args.ranked_file) 
    print('====== Finish ======')