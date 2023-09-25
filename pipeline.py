class Pipeline:
    """
    Pipeline has the following steps:
    1. Load the wav-data of multi-user speech
    2. Chunk data into windows of 5 sec with 0.5 sec overlap
    3. In each window segment speakers: for each speaker predict probability that he speaks in each of 16 ms frames
    4. For each speaker within each 5 sec window get all the 16 ms chunks where he speaks, concatenate them into one
    and get an embedding of this chunk from neural network embedding model
    5. Clusterize all the local embeddings to get global label for each speaker in the whole wav-file.
    """
    def __init__(self):
        ...

    def run_vad(self):
        """
        Run voice activity detection on wav-file.
        It may be binary or multi-class model:
        1. Binary model: predict probability that each 16 ms frame contains speech
        2. Multi-class model: predict probability for each speaker that each 16 ms frame contains her speech
        """
        ...

    def embed(self):
        """
        Get embeddings for each speaker in each 5 sec window.
        """
        ...

    def run_clustering(self):
        ...


def main():
    ...
