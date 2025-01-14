import os
import logging
import requests
from langchain_openai import ChatOpenAI

LLM_MODEL_NAME_DICT = {
    "Qwen": "Qwen/Qwen2.5-32B-Instruct",
    "Xiaobei": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2": "Qwen/Qwen2.5-72B-Instruct",
    "Aliyun": "qwen-turbo", 
}

# Please change URL to your actural URL
LLM_BASE_URL_DICT = {
    "Qwen": "http://localhost:8284/v1", 
    "Xiaobei": "http://localhost:8285/v1", 
    "Qwen2": "http://localhost:8286/v1",
    "Aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

qwen_url = 'http://localhost:8286/qwen'
xiaobei_url = 'http://localhost:8286/xiaobei'

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class CustomLanguageModel():
    def __init__(self, llm_type: str, use_local: bool = True):
        if os.getenv('OPENAI_API_KEY') is None:
            os.environ['OPENAI_API_KEY'] = 'your open-ai api key here'
        
        self.llm_type: str = llm_type
        self.model = ChatOpenAI(
            model=LLM_MODEL_NAME_DICT[llm_type],
            base_url=LLM_BASE_URL_DICT[llm_type],
        )

        self.max_retry_times = 5
        self.use_local = use_local
        if use_local:
            self.agent = None

    def __call__(self, query, config=None, stop=None):
        return self.model.invoke(query, config=config, stop=stop).content
    
    def invoke(self, query, config=None, stop=None):
        return self.model.invoke(query, config=config, stop=stop)
    
    def batch(self, prompts):
        responses = self.model.batch(prompts)
        return [response.content for response in responses]
    
    def generate(
        self, 
        query, 
        max_new_tokens=4096, 
        use_logprob=False, 
        use_attention=False,
        use_entropy=False,
        use_logits=False,
    ):
        if self.use_local:
            if self.agent is None:
                from microservice import BasicGenerator, LoraGenerator
                from microservice.config import model_path, lora_model_path
                
                if self.llm_type == 'Qwen':
                    self.agent = BasicGenerator(model_path)
                elif self.llm_type == 'Xiaobei':
                    self.agent = LoraGenerator(model_path, lora_model_path)
                else:
                    raise NotImplementedError
            
            return self.agent.generate_all(
                query, 
                max_new_tokens=max_new_tokens, 
                solver="max", 
                use_logprob=use_logprob, 
                use_attention=use_attention, 
                use_entropy=use_entropy, 
                use_logits=use_logits
            )
        else:
            if self.llm_type.startswith('Qwen'):
                url = qwen_url
            elif self.llm_type == 'Xiaobei':
                url = xiaobei_url
            else:
                raise NotImplementedError
            
            headers = {
                'Content-Type': 'application/json'
            }
            params = {
                'content': query,
                'max_new_tokens': max_new_tokens,
                'use_logprob': use_logprob,
                'use_attention': use_attention,
                'use_entropy': use_entropy,
                'use_logits': use_logits,
            }

            for _ in range(self.max_retry_times):
                response = requests.post(url, json=params, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    return result
                else:
                    logger.warning(f"Request failed with status code {response.status_code}: {response.text}")
            
            raise Exception(f"Failed to get a valid response, status code: {response.status_code}")            
        

if __name__ == "__main__":
    import timeit
    # llm = CustomLanguageModel("Qwen2")
    # prompt = "how to make a sandwich?"
    # print('-' * 10)
    # start_time = timeit.default_timer()
    # result = llm(prompt)
    # print(result)
    # print('[Qwen2] Use time:', timeit.default_timer() - start_time)
    
    llm = CustomLanguageModel("Qwen")
    prompt = "how to make a sandwich?"
    print('-' * 10)
    start_time = timeit.default_timer()
    result = llm.generate(prompt, use_logprob=True)
    print(result['text'])
    print('[Qwen] Use time:', timeit.default_timer() - start_time)
    
    llm = CustomLanguageModel("Xiaobei")
    prompt = "how to make a sandwich?"
    print('-' * 10)
    start_time = timeit.default_timer()
    result = llm.generate(prompt, use_logprob=True)
    print(result['text'])
    print('[Xiaobei] Use time:', timeit.default_timer() - start_time)

