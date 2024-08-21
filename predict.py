import utils
import jieba
import numpy as np
from typing import *
from sklearn.feature_extraction.text import CountVectorizer


class Classifier:
    def __init__(self, model_dir: str):
        vocab, self.p_c, self.p_w_c, self.labels = utils.load_model(model_dir)
        self.cv = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)

    def predict_text(self, text: str, top_n: int=1) -> List[Tuple[str, float]]:
        seg = " ".join(utils.jieba_segment(text, mode="search"))
        text_vec = self.cv.transform([seg])
        log_p_d_c = text_vec @ np.log(self.p_w_c)
        log_p_c_d = np.log(self.p_c).reshape(-1, 1) + log_p_d_c.T
        prob = utils.softmax(log_p_c_d)
        top_n_index = prob[:, 0].argsort()[::-1][:top_n]
        return [(self.labels[index], prob[:, 0][index]) for index in top_n_index]


if __name__ == "__main__":
    cls = Classifier("resources/model/new_model")
    res = cls.predict_text("Let's listen to some music", top_n=3)
    print(res)
