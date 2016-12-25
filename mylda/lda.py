from collections import defaultdict
from string import ascii_lowercase

import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from .settings import alpha, beta, K, iter_max, dict_file as default_dict
from .utils import n2s
from .sample import _sample


class Dictionary:
    """Dictionary used to map word to token id or conversely."""

    def __init__(self, corpus=None, dict_file=None, token_list=None,
                 min_tf=None, min_df=None, max_dict_len=None, stem=False):
        """Pass corpus as a sequence of list consisting of words, or a
        dict_file in which a line contains a single word, or a list of
        tokens."""
        self.stem = stem
        # Create tokenizer
        self.tokenizer = RegexpTokenizer(r"[a-z]+")

        # Create English stop words list
        self.en_stop = get_stop_words('en')
        self.en_stop += list(ascii_lowercase)

        # Create p_stemmer of class PorterStemmer
        self.p_stemmer = PorterStemmer()

        if corpus is not None:
            self._from_corpus(corpus, min_tf=min_tf, min_df=min_df,
                              max_dict_len=max_dict_len, stem=stem)
        elif dict_file is not None:
            token_list = open(dict_file).read().split("\n")
            self._from_tokens_list(token_list)
        elif token_list is not None:
            self._from_tokens_list(token_list)

    def _from_corpus(self, corpus, min_tf, min_df, max_dict_len, stem):
        self._tokenTf = defaultdict(int)
        self._tokenDf = defaultdict(int)

        for doc in corpus:
            raw = doc.lower()
            tokens = self.tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if i not in self.en_stop]
            if self.stem:
                stemmed_tokens = [self.p_stemmer.stem(i)
                                  for i in stopped_tokens]
                words = stemmed_tokens
            else:
                words = stopped_tokens
            for word in words:
                self._tokenTf[word] += 1
            words = set(words)
            for word in words:
                self._tokenDf[word] += 1

        selected_tokens = []

        if min_tf is not None:
            self._min_tf = min_tf
            for token in self._tokenTf.keys():
                if self._tokenTf[token] >= min_tf:
                    selected_tokens.append(token)
        else:
            selected_tokens = list(self._tokenTf.keys())

        if min_df is not None:
            selected_tokens = [x for x in selected_tokens
                               if self._tokenDf[x] >= min_df]

        if max_dict_len is not None and max_dict_len < len(selected_tokens):
            _ = sorted(selected_tokens, key=lambda x: -self._tokenTf[x])
            selected_tokens = _[:max_dict_len]

        self.token_list = selected_tokens
        self._token2id = {token: id for id, token
                          in enumerate(selected_tokens)}

    def _from_tokens_list(self, token_list):
        self.token_list = token_list
        self._token2id = {token: id for id, token
                          in enumerate(self.token_list)}

    def id2token(self, tokenId):
        return self.token_list[tokenId]

    def token2id(self, tokenStr):
        if tokenStr in self._token2id:
            return self._token2id[tokenStr]
        return None

    def doc2tokens(self, doc):
        raw = doc.lower()
        tokens = self.tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if i not in self.en_stop]
        if self.stem:
            stemmed_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]
            words = stemmed_tokens
        else:
            words = stopped_tokens
        tokenids = [self.token2id(token)
                    for token in words if self.token2id(token)]
        return tokenids

    def save(self, fname="dictionary.txt"):
        with open(fname, "wt") as f:
            for token in self.token_list:
                f.write(token + "\n")


class LDA:
    """A collapsed gibbs-sampling based implementation of LDA model."""

    def __init__(self, K=K, alpha=alpha, beta=beta, n_early_stop=10,
                 iter_max=iter_max, dict_file=None, use_default_dict=False,
                 dictionary=None):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_early_stop = n_early_stop
        self.iter_max = iter_max
        self.dict_file = dict_file
        self.dictionary = dictionary
        if use_default_dict:
            self.dict_file = default_dict

    def _load_data(self, n_mt=None, content_list=None, tokenid_list=None,
                   min_tf=None, min_df=None, max_dict_len=None, stem=True):
        """You can pass data by either doc-token count matrix n_mt or a
        sequence of list consisting of words content_lis(preferred)."""
        if self.dictionary is not None:
            pass
        elif self.dict_file is not None:
            self.dictionary = Dictionary(dict_file=self.dict_file)
        elif content_list is None:
            raise Exception("You should pass a dictionary file since you"
                            " haven't create a dictionary using corpus")

        if n_mt is not None:
            self.M, self.T = n_mt.shape
            self.N = n_mt.sum()
            self.w_mi = [n2s(n_m) for n_m in n_mt]
            return

        self.N = 0

        if tokenid_list is not None:
            self.T = len(self.dictionary.token_list)
            self.M = len(tokenid_list)
            self.w_mi = tokenid_list
            for doc in tokenid_list:
                self.N += len(doc)
            return

        if content_list is not None:
            if self.dictionary is None:
                print("Generating dictionary...")
                self.dictionary = Dictionary(corpus=content_list,
                                             min_tf=min_tf,
                                             min_df=min_df,
                                             max_dict_len=max_dict_len,
                                             stem=stem)
            self.T = len(self.dictionary.token_list)
            self.M = len(content_list)
            self.w_mi = [[] for x in range(self.M)]
            for m, doc in enumerate(content_list):
                if m % 20 == 0:
                    print("read doc %d/%d" % (m, self.M))
                self.w_mi[m] = self.dictionary.doc2tokens(doc)
                self.w_mi[m] = np.array(self.w_mi[m], dtype=np.int32)
                self.N += len(self.w_mi[m])

    def _initialize(self):
        """After data is loaded, we have to assign random topic token for each
        word in the doc, and prepare some of the neccesary statistics"""

        # the topic token of i-th word in m-th doc
        self.z_mi = [[] for m in range(self.M)]
        # the count of word token t in k-th topic
        self.n_kt = np.zeros((self.K, self.T), dtype=np.int32)
        # the count of words assigned to k-th topic in m-th doc
        self.n_mk = np.zeros((self.M, self.K), dtype=np.int32)
        for m in range(self.M):
            if m % 20 == 0:
                print("initialize doc %d/%d" % (m, self.M))
            I = len(self.w_mi[m])
            self.z_mi[m] = np.random.randint(self.K, size=I).astype(np.int32)
            for i, z in enumerate(self.z_mi[m]):
                self.n_mk[m, z] += 1
                self.n_kt[z, self.w_mi[m][i]] += 1
        self._list2array()

    def _list2array(self):
        """Convert neccesary statistic variables to numpy one-dimensional
         array, with the purpose of following cython acceleration """
        self.W = np.zeros(self.N, dtype=np.int32)
        self.Z = np.zeros(self.N, dtype=np.int32)
        self.N_m = np.zeros(self.M, dtype=np.int32)
        self.I_m = np.zeros(self.M, dtype=np.int32)
        n1 = 0
        for m in range(self.M):
            self.N_m[m] = len(self.w_mi[m])
            n2 = n1 + self.N_m[m]
            self.W[n1:n2] = self.w_mi[m]
            self.Z[n1:n2] = self.z_mi[m]
            self.I_m[m] = n1
            n1 = n2
        self.n_kt_sum = np.sum(self.n_kt, axis=1, dtype=np.int32)

    def _phi(self):
        """return the infered topics 'phi'. """
        smoothed = self.n_kt + self.beta
        phi = smoothed / np.c_[smoothed.sum(axis=1)]
        return phi

    def fit(self, inputData, min_tf=10, min_df=None,
            max_dict_len=None, stem=False):
        if isinstance(inputData, np.ndarray) and len(inputData.shape) > 1:
            self._load_data(n_mt=inputData)
        elif isinstance(inputData[0][0], (int, np.integer)):
            self._load_data(tokenid_list=inputData)
        elif isinstance(inputData[0][0], str):
            self._load_data(content_list=inputData, min_tf=min_tf,
                            min_df=min_df, max_dict_len=max_dict_len,
                            stem=stem)
        else:
            raise Exception("Input type not supported!")

        self._initialize()

        iter_num = self.iter_max
        num_change_min = float("inf")
        num_not_min = 0
        num_z_changes = [None, ] * iter_num
        for i in range(iter_num):
            num_z_change = _sample._train(self.n_mk, self.n_kt,
                                          self.n_kt_sum, self.W, self.Z,
                                          self.N_m, self.I_m,
                                          self.alpha, self.beta)
            num_z_changes[i] = num_z_change
            print("iter %d/%d.\t z_change:%d" % (i, iter_num, num_z_change))

            if num_z_change < num_change_min:
                num_change_min = num_z_change
                num_not_min = 0
            else:
                num_not_min += 1

            if num_not_min >= self.n_early_stop:
                break
        return num_z_changes[:i+1]

    def topic_top_words(self, topNum=30):
        """Return a sequence of list of top-words in each topic"""
        result = [[] for x in range(self.K)]
        for k in range(self.K):
            topWords = self.n_kt[k, :].argsort()[::-1][:topNum]
            result[k] = [self.dictionary.id2token(t) for t in topWords]
        return result

    def show_topic(self, topNum=30):
        """Return a str format representation of topics"""
        topics = self.topic_top_words(topNum)
        str_repr = ""
        for topic in topics:
            current_line = " ".join(topic)
            print(current_line)
            print("=" * 30)
            str_repr += (current_line + "\n" + "=" * 30 + "\n")
        return str_repr
