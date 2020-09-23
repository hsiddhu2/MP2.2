import math
import sys
import time

import metapy
import pytoml


class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """

    def __init__(self, some_param=1.0):
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html

        avg_dl = {float} 87.1785736084
        corpus_term_count = {long} 373
        d_id = {long} 1
        doc_count = {long} 251
        doc_size = {long} 112
        doc_term_count = {long} 1
        doc_unique_terms = {long} 63
        num_docs = {long} 1400
        query_length = {float} 10.0
        query_term_weight = {float} 1.0
        t_id = {long} 1618
        total_terms = {long} 122050
        """

        tfn = sd.doc_term_count * math.log2(1.0 + sd.avg_dl / sd.doc_size)
        middle_term = tfn / (tfn + self.param)
        sub_term = (sd.num_docs + 1) / (sd.corpus_term_count + 0.5)
        return sd.query_term_weight * middle_term * math.log2(sub_term)

        # return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    return InL2Ranker(some_param=1.0)

    # return metapy.index.JelinekMercer()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            print("Query {} average precision: {}".format(query_num + 1, avg_p))
    print("Mean average precision: {}".format(ev.map()))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
