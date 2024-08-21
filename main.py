import utils
import logging
import numpy as np
from typing import *
from tqdm import tqdm
from category import Category
from sklearn.feature_extraction.text import CountVectorizer

# This function traverses the document, pre-annotates based on the initial keywords, and generates a document word frequency matrix
def preliminary_labeling(category_tree: Category, segs: List[List[str]]):
    # The default token_pattern filters out words
    cv = CountVectorizer(analyzer="word", max_df=0.8, min_df=0.00001, token_pattern=r"(?u)\b\w+\b")
    logging.info("Initializes the document word frequency matrix")
    document_vectors = cv.fit_transform([" ".join(seg) for seg in segs])  # csr_matrix
    vocabulary = cv.vocabulary_
    logging.info("vocabulary size: {}".format(len(vocabulary)))
    logging.info("Document prelabeling")
    for i, seg in tqdm(enumerate(segs)):
        for word in seg:
            category = category_tree.find_category_by_word(word)
            if category is not None:
                category.add_document(i)
                break
    return vocabulary, document_vectors

# Initializes the parameters of the Bayesian model, including category prior probability, document conditional probability, and word conditional probability.
def init_bayes_model(category_tree: Category, documents_size: int, vocab_size: int):
    """
    return: P(C) -> (category_size, )
             P(C|D) -> (category_size, documents_size)
             P(W|C) -> (vocab_size, category_size)
    """
    category_list = category_tree.get_category_list()
    category_size = len(category_list)
    category_prior_probability = np.zeros(category_size)  # Class prior probability P(C)
    category_document_cond_probability = np.zeros(([documents_size, category_size]))  # Document Conditional probability P(C|D)

    # Initialize P(C) and P(C|D) according to the pre-labeled result
    logging.info("Parameter initialization")
    for c, category in tqdm(enumerate(category_list)):
        category_path = category.split("/")
        category_documents = category_tree.find_category(category_path).get_documents()
        for document_index in category_documents:
            category_document_cond_probability[document_index, c] = 1.0
        category_prior_probability[c] = (1.0 + len(category_documents)) / (category_size + documents_size)  # using Laplace smooth

    category_document_cond_probability = category_document_cond_probability.T
    word_category_cond_probability = np.zeros([vocab_size, len(category_list)])
    logging.info("Prelabeled scale: {}/{}".format(int(category_document_cond_probability.sum()), documents_size))

    return category_prior_probability, category_document_cond_probability, word_category_cond_probability


def maximization_step(document_vectors, p_c, p_c_d, p_w_c):
    #After updating P(C|D) in E-step, update P(W|C) and P(C) in M-step
    logging.info("Horizontal M-step")
    category_vectors = p_c_d @ document_vectors  # shape=(category_size, vocab_size)
    category_size = p_c.shape[0]
    documents_size = document_vectors.shape[0]
    vocab_size = document_vectors.shape[1]
    for c in tqdm(range(category_size)):
        category_vectors_sum = category_vectors[c].sum()
        for v in range(vocab_size):
            p_w_c[v, c] = (1 + category_vectors[c, v]) / (vocab_size + category_vectors_sum)
    for c in range(category_size):
        p_c[c] = (1.0 + p_c_d[c].sum()) / (category_size + documents_size)

# After the E step update (P(C|D)), the M step update (P(W|C)) and (P©). Class vectors are computed by matrix multiplication and Laplacian smoothing is used.
def maximization_step_with_shrinkage(category_tree: Category, document_vectors, p_c, p_c_d, p_w_c, p_w_c_k, lambda_matrix, beta_matrix, iter: int):
    documents_size, vocab_size = document_vectors.shape
    category_size, lambda_size = lambda_matrix.shape
    category_list = category_tree.get_category_list()
    # vertical M
    if iter > 0:
        shrinkage_maximization_step(lambda_matrix, beta_matrix, p_c_d)
    # horizontal M
    # update P^{α}(w|c)
    logging.info("Horizontal M-step")
    for c in tqdm(range(category_size)):
        category_path = category_list[c].split("/")
        dep_list = []
        category_depth = len(category_path)
        for k in range(category_depth):
            dep_list.append(category_list.index("/".join(category_path)))
            category_vectors = p_c_d[dep_list] @ document_vectors
            if category_vectors.ndim == 1:
                category_vectors = category_vectors.reshape(1, -1)
            category_vector_hierarchy = category_vectors.sum(axis=0)
            category_vector_hierarchy_sum = category_vector_hierarchy.sum()
            for v in range(vocab_size):
                p_w_c_k[v, c, k] = (1.0 + category_vector_hierarchy[v]) / (vocab_size + category_vector_hierarchy_sum)
            category_path = category_path[:-1]
    category_vector_root = document_vectors.sum(axis=0)
    category_vector_root_sum = document_vectors.sum()
    for v in range(vocab_size):
        p_w_c_k[v, :, -2] = (1.0 + category_vector_root[0, v]) / (vocab_size + category_vector_root_sum)  # category_vector_root.ndim=2
    p_w_c_k[:, :, -1] = 1.0 / vocab_size
    # update p_w_c
    for v in range(vocab_size):
        p_w_c[v] = (lambda_matrix * p_w_c_k[v]).sum(axis=1)
    # update p_c
    for c in range(category_size):
        p_c[c] = (1 + p_c_d[c].sum()) / (category_size + documents_size)

# After the M step update (P(W|C)) and (P(C)), the E step update (P(C|D)). Change the multiplicative to cumulative by calculating the logarithm.
def expectation_step_with_shrinkage(document_vectors, p_c, p_w_c, p_w_c_k, lambda_matrix, beta_matrix):
    # vertical E
    shrinkage_expectation_step(document_vectors, lambda_matrix, beta_matrix, p_w_c_k)
    # horizontal E
    logging.info("Horizontal E-step")
    log_p_d_c = document_vectors @ np.log(p_w_c)  # shape=(documents_size, category_size)
    log_p_c_d = np.log(p_c).reshape(-1, 1) + log_p_d_c.T  # shape=(category_size, documents_size)
    return utils.softmax(log_p_c_d)

# Initializes the parameters of the Shrink step.
def hierarchical_shrinkage_init(category_tree: Category, document_vectors):
    logging.info("Initialize the shrinkage parameter")
    max_depth = Category.get_max_depth(category_tree)
    category_list = category_tree.get_category_list()
    category_size = len(category_list)
    lambda_size = max_depth + 2
    lambda_matrix = np.zeros([category_size, lambda_size])
    for c, path in enumerate(category_list):
        category_node = category_tree.find_category(path.split("/"))
        depth = category_node.get_depth()
        init_lambda_val = 1.0 / (depth + 2)
        for k in range(depth):
            lambda_matrix[c, k] = init_lambda_val
        lambda_matrix[c, max_depth] = init_lambda_val
        lambda_matrix[c, max_depth+1] = init_lambda_val
    # init β
    documents_size, vocab_size = document_vectors.shape
    beta_matrix = np.zeros([documents_size, category_size, lambda_size])
    # init P^{α}(W|C)
    p_w_c_k = np.zeros([vocab_size, category_size, lambda_size])
    return lambda_matrix, beta_matrix, p_w_c_k


def shrinkage_maximization_step(lambda_matrix, beta_matrix, p_c_d):
    # update λ
    logging.info("Vertical M-step")
    documents_size, category_size, lambda_size = beta_matrix.shape
    for c in tqdm(range(category_size)):
        norm_val = p_c_d[c].sum()
        for k in range(lambda_size):
            lambda_matrix[c, k] = beta_matrix[:, c, k] @ p_c_d[c]
            lambda_matrix[c, k] /= norm_val


def shrinkage_expectation_step(document_vectors, lambda_matrix, beta_matrix, p_w_c_k):
    # update β
    logging.info("Vertical E-step")
    documents_size = document_vectors.shape[0]
    document_vectors_nonzero = document_vectors.nonzero()
    document_vectors_nonzero_count = np.bincount(document_vectors_nonzero[0], minlength=documents_size)
    for d, v in tqdm(zip(*document_vectors_nonzero)):
        p_w_c_alpha = lambda_matrix * p_w_c_k[v]  # shape = (category_size, lambda_size)
        p_w_c_alpha = p_w_c_alpha / p_w_c_alpha.sum(axis=1).reshape(-1, 1)
        beta_matrix[d] += p_w_c_alpha

    for d in range(documents_size):
        if document_vectors_nonzero_count[d] > 0:
            beta_matrix[d] /= document_vectors_nonzero_count[d]


def main(word_file: str, corpus_file: str, result_file: str, model_save_path=None, max_iters=5):
    """
    param word_file: Keyword file path
    param corpus_file: Path of the sample file to be classified
    param result_file: Category result saving path
    param model_save_path: Saving path of model parameters
    param max_iters: Iteration rounds
    return:
    """
    category_tree = utils.load_seed_keywords(word_file)
    datas = utils.load_data(corpus_file)
    segs = utils.word_segment(datas)
    vocabulary, document_vectors = preliminary_labeling(category_tree, segs)
    del segs
    p_c, p_c_d, p_w_c = init_bayes_model(category_tree, documents_size=len(datas), vocab_size=len(vocabulary))
    lambda_matrix, beta_matrix, p_w_c_k = hierarchical_shrinkage_init(category_tree, document_vectors)
    for _i in range(max_iters):
        logging.info("EM iterative advance: {}/{}".format(_i + 1, max_iters))
        maximization_step_with_shrinkage(category_tree, document_vectors, p_c, p_c_d, p_w_c, p_w_c_k, lambda_matrix, beta_matrix, _i)
        p_c_d = expectation_step_with_shrinkage(document_vectors, p_c, p_w_c, p_w_c_k, lambda_matrix, beta_matrix)

    category_list = category_tree.get_category_list()
    fw = open(result_file, "w", encoding="utf-8")
    for i in range(len(datas)):
        prob = p_c_d[:, i]
        predict_category = category_list[prob.argmax()]
        fw.write(datas[i] + "\t" + predict_category + "\n")
    fw.close()

    if model_save_path is not None:
        utils.save_model(model_save_path, vocabulary, p_c, p_w_c, category_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(word_file="resources/dict/words.txt",
         corpus_file="resources/cropus/data.txt",
         result_file="resources/cropus/data_result.txt",
         model_save_path="resources/model/new_model",
         max_iters=5)
