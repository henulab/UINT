import os
import jieba
import pickle
import logging
import collections
import numpy as np
from typing import *
from tqdm import tqdm

from category import Category


def load_seed_keywords(keywords_file: str) -> Category:
    category = Category("ROOT")
    category_list = []
    with open(keywords_file, "r", encoding="utf-8") as fr:
        for line in fr:
            categories, words = line.strip().split("###")
            category_list.append(categories)
            categories = categories.split("/")
            words = set(words.split("|"))
            category.add_category(categories).set_keywords(words)
    category.set_category_list(category_list)
    logging.info("Classification tree: {}".format(category))
    return category


def load_data(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as fr:
        datas = [line.strip() for line in fr]
    return datas


def word_segment(datas: List[str], mode: str="search") -> List[List[str]]:
    segs = []
    for data in tqdm(datas, desc="Document set presegmentation"):
        document = data.split("_!_", 3)[-1]
        document = document.replace("_!_", " ")
        seg = jieba_segment(document, mode=mode)
        segs.append(seg)
    logging.info("Before participle: {}".format(datas[0]))
    logging.info("After participle: {}".format(segs[0]))
    logging.info("If the field you fetched is different from what you expected, modify the utils.word_segment to ensure that it matches the format of the dataset")
    return segs


def jieba_segment(text: str, mode: str) -> List[str]:
    seg = list(jieba.tokenize(text, mode=mode))
    # build DAG
    graph = collections.defaultdict(list)
    for word in seg:
        graph[word[1]].append(word[2])

    def dfs(graph: Dict[int, List[int]], v: int, seen: List[int]=None, path: List[int]=None):
        if seen is None:
            seen = []
        if path is None:
            path = [v]
        seen.append(v)
        paths = []
        for t in graph[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(dfs(graph, t, seen, t_path))
        return paths

    longest_path = sorted(dfs(graph, 0), key=lambda x: len(x), reverse=True)[0]
    longest_seg = [text[longest_path[i]: longest_path[i + 1]] for i in range(len(longest_path) - 1)]
    return longest_seg


def softmax(x):
    norm_x = x - x.max(axis=0)
    return np.exp(norm_x) / np.exp(norm_x).sum(axis=0)


def save_model(model_dir: str, vocabulary: Dict[str, int], p_c: np.ndarray, p_w_c: np.ndarray, labels: List[str]):
    if not model_dir.endswith("/"):
        model_dir += "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir + "vocab.pkl", "wb") as fw:
        pickle.dump(vocabulary, fw)
    np.save(model_dir + "category_prob.npy", p_c)
    np.save(model_dir + "word_prob.npy", p_w_c)
    with open(model_dir + "labels.txt", "w", encoding="utf-8") as fw:
        for l in labels:
            fw.write(l + "\n")
    logging.info("Model saved successfully")


def load_model(model_dir: str):
    if not model_dir.endswith("/"):
        model_dir += "/"
    with open(model_dir + "vocab.pkl", "rb") as fr:
        vocabulary = pickle.load(fr)
    p_c = np.load(model_dir + "category_prob.npy")
    p_w_c = np.load(model_dir + "word_prob.npy")
    with open(model_dir + "labels.txt", "r", encoding="utf-8") as fr:
        labels = [line.strip() for line in fr]

    return vocabulary, p_c, p_w_c, labels
