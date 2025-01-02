if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Base
from microservice import DocumentSearch


class BasicRAG(Base):
    def __init__(self, args):
        super().__init__(args)

        self.retriever = DocumentSearch()

    def retrieve(self, query):
        return self.retriever(query)
    
    def inference(self, query):
        docs = self.retrieve(query)
        prompt = f"{query}\n\ncontext:\n{docs}"
        response = self.model(prompt)
        return response

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    args.model_name = "Qwen2"
    rag = BasicRAG(args)
    print(rag.inference("SIgA主要存在于"))