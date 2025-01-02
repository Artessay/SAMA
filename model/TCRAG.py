import re
import json5
import logging

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Base
from model.Tools import Tools
from model.system_score import (
    calculate_cuct_from_entropy,
    calculate_perplexity_from_logits,
    calculate_low_probablity_from_logprobs,
    calculate_low_attention_from_attentions,
)

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
TOOL_DESC = """{name_for_model}: 使用 {name_for_human} 这个API交互. 那么这个 {name_for_human} API 怎么使用呢? {description_for_model} 参数: {parameters} 格式需要是JSON对象."""
# REACT_PROMPT_EN = """Answer the following questions as best you can. You have access to the following tools:

# {tool_descs}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!
# """

REACT_PROMPT = """
你的知识不一定正确，所以你一定要用提供的工具来思考，并给出用户答案。
你有以下工具可以使用:
{tool_descs}

请【严格按照】提供的思维方式来思考，所有的关键词都要输出并且不得漏字，例如Action，Action Input，Observation等。
你的输出至少要包含Action或者Final Answer中的一个。
```
Question: 用户的提问或者观察到的信息，
Thought: 你应该思考该做什么，是根据工具的结果来回答问题，还是决定使用什么工具。
Action: 需要使用的工具，必须是[{tool_names}]中的一个，不要添加其他任何多余的字符！只能输出工具的名字！！
Action Input: 传入工具的内容
Observation: 工具给出的答案（不是你生成的）
... (这里的 Thought/Action/Action Input/Observation 可以是零次也可以被重复许多次)
Thought: 通过工具给出的答案，你是否能回答Question。
Final Answer是你的答案，且Final Answer必须详尽且有意义

现在，我们开始！一定要使用工具回答问题，请至少使用四次工具辅助你回答问题！在利用工具时，请结合自己的判断去检索部分你不知道的知识！
注意，如果你对很多问题不是很清楚时或者不清楚最后答案的时候，你可以分别调用多次工具（尤其是分别调用多次知识检索来提升可信度）！最后回答问题请回答选项！
用户开始提问：
"""

# note： Backtrack: 当上一个动作输出大量词汇你需要总结，或当上一个动作的结果对你任务无意义时，请输出自己的总结或重新执行思考结果

POP_REACT_PROMPT = """
你的知识不一定正确，所以你一定要用提供的工具来思考，并给出用户答案。
你有以下工具可以使用:
{tool_descs}

请【严格按照】提供的思维方式来思考，所有的关键词都要输出并且不得漏字，例如Thought, Action，Action Input，Observation, Backtrack, Summary等。
你的输出至少要包含Action, Backtrack或者Final Answer中的一个。
为了增强回答的可信程度，请一定多使用Backtrack来增强自己的错误反思能力，不要直接得到答案。
```
Question: 用户的提问或者观察到的信息，
Thought: 你应该思考该做什么，是根据工具的结果来回答问题，还是决定使用什么工具。
Action: 需要使用的工具，必须是[{tool_names}]中的一个，不要添加其他任何多余的字符和符号！只能输出工具的名字！！
Action Input: 传入工具的内容
Observation: 工具给出的答案（不是你生成的）
Backtrack: 当上一个动作输出大量词汇你需要总结，或当上一个动作的结果对你任务无意义你需要重新执行时，请输出自己的知识的详细总结结果或重新执行反思结果
... (这里的 Thought/Action/Action Input/Observation/Backtrack 可以是零次也可以被重复许多次)
Thought: 通过工具给出的答案，你是否能回答Question。
Final Answer是你的答案，且Final Answer必须详尽且有意义

现在，我们开始！一定要使用工具回答问题，请至少使用四次工具辅助你回答问题！在利用工具时，请结合自己的判断去检索部分你不知道的知识！
注意，如果你对很多问题不是很清楚时或者不清楚最后答案的时候，你可以分别调用多次工具（尤其是分别调用多次知识检索来提升可信度）！最后回答问题请回答选项！
用户开始提问：
"""


class TCRAG(Base):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        # hyper-parameter
        rc_max_react_loop: int = 8
        topK: int = 4
        sigma: float=1.2

        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.rc_max_react_loop = rc_max_react_loop
        self.topK = topK
        assert self.topK <= self.rc_max_react_loop
        self.message_list = []
        self.state_value_list = []
        self.INIT_STATE_VALUE = 1e5
        self.sigma = sigma                  # threshold

    def inference(self, query):
        response, _, abs_trace = self.whitebox_pop_react_executor(query)
        logger.info(f"trace:\n{abs_trace}")
        return response

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = POP_REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        # 如果大模型同时输出了很多action，那么应该以第一个为主，而不是最后一个
        # i = text.rfind('\nAction:')
        # j = text.rfind('\nAction Input:')
        # k = text.rfind('\nObservation:')
        i = text.find('Action:')
        j = text.find('Action Input:')
        k = text.find('Observation:')

        # if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if -1 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + 'Observation:'  # Add it back.
            # k = text.rfind('\nObservation:')
            k = text.find('Observation:')
            plugin_name = text[i + len('Action:'): j].strip()
            plugin_args = text[j + len('Action Input:'): k].strip()
            text = text[:k]
        # 错误处理：去除使用/再次使用等中文字符的出现，之后这个工具需要更改
        plugin_name = re.sub(r'[^a-zA-Z_]', '', plugin_name)
        return plugin_name, plugin_args, text

    def parse_latest_final_answer(self, text):
        i = text.rfind('Final Answer:')
        text = text[i:]
        return text

    def call_plugin(self, plugin_name: str, plugin_args: str):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.Web_RAG(**plugin_args)
        elif plugin_name == 'HypothesisOutput':
            return '\nObservation:' + self.tool.HypothesisOutput(**plugin_args)
        elif plugin_name == 'MedicalNER':
            return '\nObservation:' + self.tool.MedicalNER(**plugin_args)
        elif plugin_name == 'DOC_RAG':
            return '\nObservation:' + self.tool.DOC_RAG(**plugin_args)
        # elif plugin_name == 'WIKI_RAG':
        #     return '\nObservation:' + self.tool.WIKI_RAG(**plugin_args)
        elif plugin_name == 'KG_RAG':
            return '\nObservation:' + self.tool.KG_RAG(**plugin_args)
        elif plugin_name == 'Baike_RAG':
            return '\nObservation:' + self.tool.Baike_RAG(**plugin_args)
        elif plugin_name == 'Web_RAG':
            return '\nObservation:' + self.tool.Web_RAG(**plugin_args)
        elif plugin_name == 'KnowledgeOrganize':
            return '\nObservation:' + self.tool.KnowledgeOrganize(**plugin_args)
        elif plugin_name == 'Filter':
            return '\nObservation:' + self.tool.Filter(**plugin_args)
        else:
            return '\nObservation:' + f"Plugin {plugin_name} not found"

    def renew_message_list(self, ):
        self.message_list = []
        self.absolute_message_list = []
        self.state_value_list.append(self.INIT_STATE_VALUE)

    def message_list_to_str(self, ):
        return_str = ""
        if self.message_list == []:
            return "No messages"
        else:
            for message in self.message_list:
                return_str += '\n' + str(message)
            return return_str

    def absoulte_message_list_to_str(self, ):
        return_str = ""
        if self.absolute_message_list == []:
            return "No messages"
        else:
            for message in self.absolute_message_list:
                return_str += '\n' + str(message)
            return return_str


    def push_message(self, message):
        # 有时候一个message可能会同时包含很多个关键词，因此需要用正则表达式分一下
        keywords = ["Question", "Thought", "Action", "Action Input", "Backtrack", "Observation", "Final Answer"]
        pattern = '|'.join([f'({keyword}\\s*:)' for keyword in keywords])
        parts = re.split(pattern, message, flags=re.IGNORECASE)

        # 移除空或只包含空白字符的字符串，同时确保部分不是 None
        parts = [part for part in parts if part and part.strip()]

        # 重构消息列表，确保不会越界
        i = 0
        while i < len(parts) - 1:
            combined = parts[i].strip() + parts[i + 1].strip()
            self.message_list.append(combined)
            if '请重新思考，或改写工具调用的方式和输入参数' in combined:      # 避免backtrack重复压栈
                pass
            else:
                self.absolute_message_list.append(combined)

            i += 2
        # 如果关键词后无文本，处理最后一个部分
        if i < len(parts):
            self.message_list.append(parts[i].strip())
            if '请重新思考，或改写工具调用的方式和输入参数' in parts[i].strip():        # 避免backtrack重复压栈
                pass
            else:
                self.absolute_message_list.append(parts[i].strip())

    def update_status_value(self, call_results, new_response):
        now_attentions = call_results["attentions"]  # N*1的attention
        now_entropies = call_results["entropies"]  # N*1的熵
        now_logprobs = call_results["logprobs"]  # N*1，代表每个token的概率
        now_logits = call_results["logits"]  # N*1，代表每个token的概率

        # 寻找正确的token对应的内容
        response_tokens = call_results["tokens"]  # 获取所有结果的token
        span_start = 0
        for i in range(len(response_tokens) - 3):
            current_token = response_tokens[i]
            print(current_token)
            if current_token == 'Thought':
                span_start = i + 2
                break
            elif current_token == 'Final':
                next_token = response_tokens[i + 1]
                if next_token == 'Answer':
                    span_start = i + 3
                break

        # debug use
        # print(span_start)
        # print(response_tokens[span_start:])

        if span_start >= len(response_tokens) - 1:  # 异常控制，避免找不到
            self.now_attentions = now_attentions
            self.now_entropies = now_entropies
            self.now_logprobs = now_logprobs
            self.now_logits = now_logits
        else:  # 能找到，直接取栈顶
            self.now_attentions = now_attentions[span_start:]
            self.now_entropies = now_entropies[span_start:]
            self.now_logprobs = now_logprobs[span_start:]
            self.now_logits = now_logits[span_start:]

    def pop_message(self, ):
        if "Backtrack" in self.top_message():
            Backtrack_info = self.message_list.pop()
            Backtrack_info += '请重新思考，或改写工具调用的方式和输入参数'
            # 当检索到回撤Backtrack关键词时，去除Thought / Action=>Observation 的栈
            if "Observation" in self.top_message():
                self.message_list.pop()     # pop observation
                if 'Action Input' in self.top_message():
                    self.message_list.pop()     # pop action input
                if 'Action' in self.top_message():
                    self.message_list.pop()     # pop action
            elif "Thought" in self.top_message():
                self.message_list.pop()  # pop Thought
            elif "Backtrack" in self.top_message():
                self.message_list.pop()  # pop Backtrack
            else:   # 为query时
                pass
            self.push_message(Backtrack_info)   # push 反馈信息入栈
        else:
            pass

    def top_message(self, ):
        return self.message_list[-1]

    def process_no_regular_output(self, message):
        # note 这里把query换成input是因为LLM有时候不遵循指令，会输出query （训练原因）
        message = message.replace("query", "input")

        keywords = ["Question", "Thought", "Action", "Action Input", "Backtrack",  "Observation", "Final Answer"]
        regular = False
        for keyword in keywords:
            if keyword in message:
                regular = True
        if not regular:
            message = 'Final Answer:' + message
        return message

    def detact_backtrack(self, message):
        # 有时候一个message可能会同时包含很多个关键词，因此需要用正则表达式分一下
        keywords = ["Question", "Thought", "Action", "Action Input", "Backtrack", "Observation", "Final Answer"]
        pattern = '|'.join([f'({keyword}\\s*:)' for keyword in keywords])
        parts = re.split(pattern, message, flags=re.IGNORECASE)

        # 移除空或只包含空白字符的字符串，同时确保部分不是 None
        parts = [part for part in parts if part and part.strip()]
        i = 0

        while i < len(parts) - 1:
            combined = parts[i].strip() + parts[i + 1].strip()
            if "Backtrack" in combined:
                break
            else:
                combined = message
            i += 2
        return combined

    def calculate_now_state_value(self, top_content=None, metric_type: float="cuct"):       # fixme here: 这是计算state value以及压栈的过程
        self.bottom_content = self.message_list[0]
        if top_content is None:
            self.top_content = self.message_list[-1]
        else:
            self.top_content = top_content

        if metric_type == "cppl":       # ppl
            # 这里需要得到logits
            total_logits_response = self.model.generate(self.bottom_content, use_logits=True)
            total_logits = total_logits_response['logits']
            status_value = calculate_perplexity_from_logits(total_logits, self.now_logits)
        elif metric_type == 'cuct':     # uncertainty
            # 这里需要得到entropies
            status_value = calculate_cuct_from_entropy(self.now_entropies)
        elif metric_type == 'low_prob':
            # 这里需要得到logprobs
            status_value = calculate_low_probablity_from_logprobs(self.now_logprobs)
        elif metric_type == 'attention':
            status_value = calculate_low_attention_from_attentions(self.now_attentions)
        else:
            pass

        logger.info(f"Status Value: {status_value}")
        self.state_value_list.append(status_value)

    def obtain_now_state(self):                 # 这是取得status栈顶的过程
        return self.state_value_list[-1]

    def backtrack_now_state(self):              # 如果有回撤，那么这里就需要backtrack
        self.state_value_list.pop()

    # note 24/08/09版本
    def whitebox_pop_react_executor(self, text):
        # todo 引入了status value
        '''
        最关键的就是executor的执行循环了，executor会始终进行如下事件循环直到 目标被解决了 或者 思考迭代次数超过了最大次数：
        根据之前已经完成的所有步骤（一个步骤包括 ReAct框架中的 Thought、Action、Observation）和 目标（用户的问题）规划出接下来的Action（使用什么工具 以及 工具的输入）
        检测是否已经达成目标，即Action是不是ActionFinish。是的话就返回结果，不是的话说明还有行动要完成
        根据Action，执行具体的工具，等待工具返回结果。工具返回的结果就是这一轮步骤的Observation
        保存当前步骤到记忆上下文，如此反复
        '''

        # 0. 组装prompt
        self.renew_message_list()  # 将消息队列先置为空，并将当前状态置于最大值

        # 1. 将user query放入message list中
        user_query = "Question:" + text
        self.push_message(user_query)

        actions_taken = 0
        while actions_taken < self.rc_max_react_loop:
            # 1. 先plan，得到新的response
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.message_list_to_str()}]
            call_results = self.model.generate(messages, use_logprob=True, use_attention=True, use_entropy=True, use_logits=True)
            ## 从call_results中取出text，attention和logprobs，entropies等信息出来
            new_response = call_results["text"]             # 文本

            # 1.5 判错条件
            new_response = self.process_no_regular_output(new_response)
            logger.info(f"response: {new_response}")

            # 2. 判断是否包含结束标识符Final Answer (LLM如果理解错了new_response也会输出final answer）
            if 'Final Answer:' in new_response:  # 忽略topK 次
                if actions_taken >= self.topK:
                    # 将结果加入消息队列中
                    new_response = self.parse_latest_final_answer(new_response)

                    # 当出现是final_answer和thought的时候，再更新系统的状态变量需要用到的tensor：
                    ## fixme 只取出new_response的内容，避免其他的影响
                    self.update_status_value(call_results, new_response)

                    self.calculate_now_state_value(top_content=new_response)

                    ## 停止条件：出现final answer且次数太多了，且小于sigma
                    if self.obtain_now_state() < self.sigma:      # 为final answer且小于simga, 模型结束, 压栈
                        self.push_message(new_response)
                        break
                    else:
                        # 这个时候由于以及压栈status了，下面又会被替换为thought，所以会被重复压栈，需要pop出去状态status
                        self.backtrack_now_state()
                # else:
                # 如果未达到指定次数，或阈值不满足要求，则替换为thought => note 这里可以做创新，step back and rethought
                new_response = new_response.replace("Final Answer:", "Thought:")

            # else:   # action / thought
            # 3. 解析可能需要使用的工具：这里的response需要将大模型回答的observation截去；保留thought
            plugin_name, plugin_args, new_response_before_detact = self.parse_latest_plugin_call(new_response)

            # 3.5 如果new_response中存在Backtrack，这时应该丢弃所有其他的动作
            new_response = self.detact_backtrack(new_response_before_detact)

            # 保留当前栈顶信息，如果当前栈顶是thought，且下一次行动是backtrack，那么状态需要回撤
            top_stack_content = self.top_message()

            self.push_message(new_response)
            if "Thought" in new_response:                             # push进站，检查thought是否存在，以及是否需要更新status
                # 当出现是final_answer和thought的时候，再更新系统的状态变量需要用到的tensor：
                ## fixme 只取出new_response的内容，避免其他的影响
                self.update_status_value(call_results, new_response)

                self.calculate_now_state_value()                      # 得到当前的state并更新系统status值

            self.pop_message()                  # 检查是否需要pop message
            # 如果pop了thought，那么就需要回撤status
            if 'Thought' in top_stack_content and "Backtrack" in new_response:
                self.backtrack_now_state() # 回撤之前的status

            # 4. 解析完工具后调用, 添加observation进入记忆中
            if plugin_name and new_response == new_response_before_detact:
                observations = self.call_plugin(plugin_name, plugin_args)
                self.push_message(observations)

            actions_taken += 1

        return self.top_message(), self.message_list_to_str(), self.absoulte_message_list_to_str()

if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    args.model_name = "Qwen"

    agent = TCRAG(args)
    reponse = agent.inference("如何用python实现一个简单的栈？")
    print(reponse)
