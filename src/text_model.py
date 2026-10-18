"""TF-IDF + logistic regression model (own module so saved models can be loaded anywhere)."""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import spelling


class TfidfLR:
    def __init__(self, use_spelling=False):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
        self.clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        self.use_spelling = use_spelling

    def _x(self, texts, fit=False):
        if fit:
            parts = [self.word.fit_transform(texts), self.char.fit_transform(texts)]
        else:
            parts = [self.word.transform(texts), self.char.transform(texts)]
        if getattr(self, "use_spelling", False):
            parts.append(csr_matrix(np.array([spelling.features(t) for t in texts])))
        return hstack(parts).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._x(texts))[:, 1]

    def spelling_weights(self):
        """Learned coefficient for each spelling feature (positive = pushes towards scam)."""
        if not getattr(self, "use_spelling", False):
            return {}
        k = len(spelling.FEATURE_NAMES)
        return dict(zip(spelling.FEATURE_NAMES, self.clf.coef_[0][-k:].round(3).tolist()))

    def top_words(self, text, k=8):
        """Word n-grams in this message with the largest positive contribution."""
        xw = self.word.transform([text])
        coefs = self.clf.coef_[0][: len(self.word.vocabulary_)]
        vocab = self.word.get_feature_names_out()
        contrib = [(vocab[j], xw[0, j] * coefs[j]) for j in xw.nonzero()[1]]
        return sorted(contrib, key=lambda t: -t[1])[:k]
