from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List

class RetrieverAgent:
    def __init__(self):
        self.persist_directory = 'db'
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        self.vectordb = Chroma(
                            persist_directory=self.persist_directory,
                            embedding_function=self.embedding
                        )

    def similarity_search(self, text, sentence_num = 5) -> List[str]:
        results = self.vectordb.similarity_search(text, 
                                             k=sentence_num)
        sentences = [doc.page_content for doc in results]
        print(sentences)

        return sentences
