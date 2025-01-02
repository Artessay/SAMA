from multiprocessing.connection import Client
from langchain_community.utilities import BingSearchAPIWrapper
import os, re
from typing import Dict, List

from microservice import WikiSearcher

"""
工具函数

- 首先要在 tools 中添加工具的描述信息
- 然后在 tools 中添加工具的具体实现

- https://serper.dev/dashboard
"""

rag_host = 'localhost'
rag_port = 63863

wiki_host = "123.57.228.132"
wiki_port = 8288

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()

        self.wiki_searcher = WikiSearcher(
            hostname=wiki_host,
            port=wiki_port,
        )

    def _tools(self):
        tools = [
            # {
            #     'name_for_human': '假设性输出模块',
            #     'name_for_model': 'HypothesisOutput',
            #     'description_for_model': '当你需要简单了解更多相关的知识时，使用这个工具可以得到一些解释，但不一定正确，需要继续使用检索医学知识工具来辅助回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识抽取模块',
            #     'name_for_model': 'MedicalNER',
            #     'description_for_model': '当需要抽取医学实体时，请使用这个工具。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要抽取的字符串形式的医学实体',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学文档知识检索模块',
            #     'name_for_model': 'DOC_RAG',
            #     'description_for_model': '使用这个工具可以得到医学文档知识，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串形式的问句',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            {
                'name_for_human': '维基百科知识检索模块',
                'name_for_model': 'WIKI_RAG',
                'description_for_model': '使用这个工具可以得到维基百科知识，请结合检索的到的部分知识来辅助你回答。',
                'parameters': [
                    {
                        'name': 'input',
                        'description': '用户询问的字符串形式的问句',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            # {
            #     'name_for_human': '医学知识图谱路径探查模块',
            #     'name_for_model': 'KG_RAG',
            #     'description_for_model': '使用这个工具可以查询两个医学实体之间的关系，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串形式的问句',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学百科知识检索模块',
            #     'name_for_model': 'Baike_RAG',
            #     'description_for_model': '使用这个工具可以查询关于某个医学实体的百科知识，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '规范名称的医学实体',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识检索模块',
            #     'name_for_model': 'Web_RAG',
            #     'description_for_model': '这是通过搜索引擎检索医学知识，请结合检索的到的部分知识来辅助你回答。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '用户询问的字符串形式的问句',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '检索的知识总结模块',
            #     'name_for_model': 'KnowledgeOrganize',
            #     'description_for_model': '当检索到的医学知识数量很多时，你可以通过将检索到的医学知识输入，然后用这个工具来做摘要总结。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要总结的字符串形式的医学知识',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # },
            # {
            #     'name_for_human': '医学知识过滤模块',
            #     'name_for_model': 'Filter',
            #     'description_for_model': '当需要过滤大量检索到的医学知识中的无关内容时，请使用这个工具。',
            #     'parameters': [
            #         {
            #             'name': 'input',
            #             'description': '需要过滤的字符串形式的医学知识',
            #             'required': True,
            #             'schema': {'type': 'string'},
            #         }
            #     ],
            # }
        ]
        return tools

    def CircumferenceMahar(self, radius: float) -> float:
        print(radius)
        return radius

    def HypothesisOutput(self, input: str) -> str:
        input = str(input)
        HO_result = send_data(input=input, function_call_type='HO')
        HO_result = str(HO_result)
        HO_result += '但是你后续必须需要检索医学知识辅助你回答，而不是一直使用假设性输出'
        return HO_result if HO_result else "无法探索性回答知识，需要再次调用或规范用户询问"

    def MedicalNER(self, input: str) -> str:
        NER_result = str(send_data(input=input, function_call_type='NER'))
        return NER_result if NER_result else "无法抽取知识，请规范用户输入"

    def DOC_RAG(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type='DOC'))
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def WIKI_RAG(self, input: str) -> str:
        RAG_result = self.wiki_searcher(query=input)
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def KG_RAG(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type='KG'))
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def Baike_RAG(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type='Baike'))
        if RAG_result == "{}":
            RAG_result = '无该医学名词的百科，请优化关键词重新查询'
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def Web_RAG(self, input: str) -> str:
        # Bing 搜索必备变量
        # 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
        # 具体申请方式请见
        # https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
        # 使用python创建bing api 搜索实例详见:
        # https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
        BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
        # 注意不是bing Webmaster Tools的api key，

        # 此外，如果是在服务器上，报Failed to establish a new connection: [Errno 110] Connection timed out
        # 是因为服务器加了防火墙，需要联系管理员加白名单，如果公司的服务器的话，就别想了GG
        BING_SUBSCRIPTION_KEY = os.environ.get('BING_SUBSCRIPTION_KEY')

        try:
            search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                          bing_search_url=BING_SEARCH_URL)
            results = search.results(input, 10)

            cleaned_results = []

            def results_to_string(results: List[Dict[str, str]]) -> str:
                """
                将处理后的搜索结果拼接为一个字符串，每条结果由标题和摘要组成。
                """
                result_strings = []
                for result in results:
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    result_string = f"标题: {title}\n摘要: {snippet}\n"
                    result_strings.append(result_string)

                # 将所有结果拼接为一个字符串，使用两个换行符分隔不同的结果
                return "\n".join(result_strings)

            for result in results:
                # 1. 提取标题
                title = result.get('title', '')
                title_cleaned = re.sub(r'<[^>]+>', '', title)  # 去除所有 HTML 标签
                title_cleaned = re.sub(r'</b>', '', title_cleaned)  # 去除 </b> 标签
                title_cleaned = re.sub(r'<b>', '', title_cleaned)  # 去除 </b> 标签
                title_cleaned = title_cleaned.strip()

                # 2. 提取摘要，并去除 HTML 标签和 </b> 标签
                snippet = result.get('snippet', '')
                snippet_cleaned = re.sub(r'<[^>]+>', '', snippet)  # 去除所有 HTML 标签
                snippet_cleaned = re.sub(r'</b>', '', snippet_cleaned)  # 去除 </b> 标签
                snippet_cleaned = re.sub(r'<b>', '', snippet_cleaned)  # 去除 </b> 标签
                snippet_cleaned = snippet_cleaned.strip()

                # 3. 提取链接，但这里我们会将其置为空字符串
                link = ''  # 去除链接

                # 4. 构造清理后的结果
                cleaned_result = {
                    'title': title_cleaned,
                    'snippet': snippet_cleaned,
                    'link': link  # 链接去掉
                }

                cleaned_results.append(cleaned_result)

            RAG_result = results_to_string(cleaned_results)

        except Exception as e:
            return f"An error occurred: {e}"
        return RAG_result if RAG_result else "无法检索到医学知识，请规范用户输入"

    def KnowledgeOrganize(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type='KO'))
        return RAG_result if RAG_result else "无法总结医学知识，请规范用户输入"

    def Filter(self, input: str) -> str:
        RAG_result = str(send_data(input=input, function_call_type='Filter'))
        return RAG_result if RAG_result else "无法过滤医学知识，请规范用户输入"


# Helper function to send data
def send_data(input, function_call_type):
    """Sends data to the server and receives the response."""
    try:
        client = Client((rag_host, rag_port))
        data_dict = {'clear': 0, 'query': input, 'function_call_type': function_call_type}
        client.send(data_dict)
        result = client.recv()  # Wait to receive data
        client.close()
        return result
    except Exception as e:
        return f"An error occurred: {e}"
    
if __name__ == '__main__':
    result = send_data('蛋糕', 'DOC')
    print(result[0][:50])