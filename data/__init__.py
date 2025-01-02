import json

def extract_questions_and_answers(data):
    extracted_data = []
    
    # 遍历 data["questions"] 列表中的每个元素
    for question_info in data["questions"]:
        question = question_info["question"]  # 提取问题
        if( question_info["answer"]["type"]=='none'):
            answer = 'none'
            context = []
        elif( question_info["answer"]["type"]=='span'):
            answer = question_info["answer"]["answer_spans"][0]["text"]  # 提取答案
            context = question_info["context"]  # 提取上下文
        elif( question_info["answer"]["type"]=='value'):
            # print(question_info["answer"])
            answer = question_info["answer"]["answer_value"]+ question_info["answer"]["answer_unit"] # 提取答案
            context = [] # 提取上下文
        elif( question_info["answer"]["type"]=='binary'):
            answer = question_info["answer"]["answer_value"] # 提取答案
            context = [] # 提取上下文  
        else:
            print(question_info["answer"])       
        # 提取相关上下文段落
        relevant_context = []
        for passage in context:
            relevant_context.append(passage["text"])
        
        # 组合问题、答案和相关上下文
        extracted_item = {
            "question": question,
            "answer": answer,
            "context": "\n".join(relevant_context)
        }
        
        extracted_data.append(extracted_item)  # 将结果添加到 extracted_data
    
    return extracted_data  # 返回提取的数据

#这个是要修改的
def load_dataset(dataset_name):
    if dataset_name == 'CMB':
        dataset_path = 'data/CMB-500.json'
    elif dataset_name == 'MMCU':
        dataset_path = 'data/MMCU_test.json'
    elif dataset_name == 'Clin':
        dataset_path = 'data/CMB-Clin.json'
    elif dataset_name == '2WikiMultihopQA':
        dataset_path = '/data1/Yangzb/data/2wikimultihopqa/data_ids_april7/dev.json'
    elif dataset_name == 'HotpotQA':
        dataset_path = '/data1/Yangzb/data/hotpotqa/hotpotqa-dev.json'
    elif dataset_name == 'StrategyQA':
        dataset_path = '/data1/Yangzb/data/strategyqa/strategyqa_train.json'
    elif dataset_name == 'IIRC':
        dataset_path = '/data1/Yangzb/data/iirc/dev.json'
    else:
        raise ValueError('Invalid dataset name')
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    if dataset_name == 'IIRC':
        data2=[]
        for item in dataset:
            extracted_data = extract_questions_and_answers(item)
            data2=data2+extracted_data
        dataset=data2
    return dataset


def get_query(data,dataset_name='CMB'):
    if dataset_name in ['CMB','MMCU','Clin']:
        data["exam_type"] = data["exam_type"]
        data["exam_class"] = data["exam_class"]
        data["exam_subject"] = data["exam_subject"]
        data["question_type"] = data["question_type"]
        data["question"] = data["question"]
        data["option_str"] = "\n".join(
            [f"{k}. {v}" for k, v in data["option"].items() if len(v) > 0 and v!=" "]
        )

        query_prompt = "以下是中国{exam_type}中{exam_class}考试的一道{exam_subject}相关的{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}"
        query = query_prompt.format_map(data)
        
        return query
    elif dataset_name == '2WikiMultihopQA':
        # Extract the question
        #数据集格式
        #{"_id": "8813f87c0bdd11eba7f7acde48001122", 
        # "type": "compositional", 
        # "question": "Who is the mother of the director of film Polish-Russian War (Film)?", 
        # "context": [["Xawery \u017bu\u0142awski", ["Xawery \u017bu\u0142awski (born 22 December 1971 in Warsaw) is a Polish film director.", "In 1995 he graduated National Film School in \u0141\u00f3d\u017a.", "He is the son of actress Ma\u0142gorzata Braunek and director Andrzej \u017bu\u0142awski.", "His second feature \"Wojna polsko-ruska\" (2009), adapted from the controversial best-selling novel by Dorota Mas\u0142owska, won First Prize in the New Polish Films competition at the 9th Era New Horizons Film Festival in Wroc\u0142aw.", "In 2013, he stated he intends to direct a Polish novel \"Z\u0142y\" by Leopold Tyrmand.", "\u017bu\u0142awski and his wife Maria Strzelecka had 2 children together:", "son Kaj \u017bu\u0142awski (born 2002) and daughter Jagna \u017bu\u0142awska (born 2009)."]], ["Snow White and the Seven Dwarfs (1955 film)", ["Snow White and the Seven Dwarfs( USA:\" Snow White\") is a 1955 German film, directed by Erich Kobler, based on the story of Schneewittchen by the Brothers Grimm."]], ["Maheen Khan", ["Maheen Khan is a Pakistani fashion and costume designer, also an award winner fashion designer for fashion labels like\" The Embroidery HouseMaheen\" and\" Gulabo\".", "She has done many national and international fashion events and shows.", "She undertook embroidery for the film Snow White and the Huntsman and television series", "The Jewel in the Crown."]], ["A Snow White Christmas", ["A Snow White Christmas is a Christmas animated television special produced by Filmation and telecast December 19, 1980, on CBS.", "It is a sequel to the fairy tale\" Snow White\", unrelated to Filmation's other sequel to\" Snow White\" titled\" Happily Ever After\"( 1990).", "The film's plot revolves around the return of the Wicked Queen, who is unexpectedly brought back to life during Christmas and casts an evil spell that freezes the entire land.", "Only the young Snow White, the daughter of the original Snow White, manages to escape and take refuge with the seven giants with her dwarf friend.", "It is now up to the giants to defeat the Queen forever and save the kingdom."]], ["Alice Washburn", ["Alice Washburn( 1860- 1929) was an American stage and film actress.", "She worked at the Edison, Vitagraph and Kalem studios.", "Her final film Snow White was her only known feature film.", "She died of heart attack in November 1929."]], ["Polish-Russian War (film)", ["Polish-Russian War", "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery \u017bu\u0142awski based on the novel Polish-Russian War under the white-red flag by Dorota Mas\u0142owska."]], ["Viktor Yeliseyev", ["Viktor Petrovich Yeliseyev( born June 9, 1950) is a Russian general, orchestra conductor and music teacher.", "He is the director of the Ministry of the Interior Ensemble, one of the two Russian Red Army Choirs."]], ["Minamoto no Chikako", ["She was the mother of Prince Morinaga."]], ["Liberty Ross", ["Liberty Lettice Lark Ross( born 23 September 1978) is an English model and actress.", "She has appeared in publications such as\" VogueHarper's Bazaari- D\", and\" Dazed& Confused\".", "She played the role of Queen Eleanor in the 2012 fantasy film\" Snow White and the Huntsman\", directed by her then- husband, Rupert Sanders.", "She is the sister of composers Atticus and Leopold Ross."]], ["Snow White and the Three Stooges", ["Snow White and the Three Stooges is the second feature film to star the Three Stooges after their 1959 resurgence in popularity.", "By this time, the trio consisted of Moe Howard, Larry Fine, and Joe DeRita( dubbed\" Curly Joe\").", "Released by 20th Century Fox, this was the trio's take on the classic fairy tale\" Snow White and the Seven Dwarfs\".", "The film was retitled Snow White and the Three Clowns in Great Britain.", "This was Walter Lang \u2018s final directing film before his retirement.", "Olympic gold medalist figure skater Carol Heiss starred as Snow White, who must flee her home after The Evil Queen, her evil stepmother, wishes her to be dead.", "Seeking refuge in the cottage of the seven dwarfs, she accidentally meets the Stooges, who are house sitting for them while they are away."]]], 
        # "entity_ids": "Q3569599_Q3570840",
        # "supporting_facts": [["Polish-Russian War (film)", 1], ["Xawery \u017bu\u0142awski", 2]], 
        # "evidences": [["Polish-Russian War", "director", "Xawery \u017bu\u0142awski"], ["Xawery \u017bu\u0142awski", "mother", "Ma\u0142gorzata Braunek"]], 
        # "answer": "Ma\u0142gorzata Braunek",
        # "evidences_id": [["Q3569599", "director", "Q3570840"], ["Q3570840", "mother", "Q274277"]], 
        # "answer_id": "Q274277"}, 
        
        data["question"] = data["question"]
        
        # # Extract the context information
        # ctxs = data["ctxs"]
        # ctx_str = "\n".join([f"{i+1}. {ctx[1]}" for i, ctx in enumerate(ctxs)])
        
        # Extract the chain of thought (CoT)
        cot = data["context"]
        data["context"] = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot)])
        
        # Extract the answer
        data["answer"] = data["answer"]
        
        # Construct the query prompt
        query_prompt = (
            "Question: {question}. \nAnswer:"#Your answer should be 'true' or 'false'. #"f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {question}. \nAnswer:'"#Your answer should be 'true' or 'false'. 
        )
        
        # Format the query prompt with the data
        query = query_prompt.format_map(data)
        return query

    elif dataset_name == 'HotpotQA':
        # Extract the question
        #数据集格式
        #[{"_id":"5a8b57f25542995d1e6f1371",
        # "answer":"yes",
        # "question":"Were Scott Derrickson and Ed Wood of the same nationality?",
        # "supporting_facts":[["Scott Derrickson",0],["Ed Wood",0]],
        # "context":[["Ed Wood (film)",["Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood."," The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau."," Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."]],["Scott Derrickson",["Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer."," He lives in Los Angeles, California."," He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""]],["Woodson, Arkansas",["Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States."," Its population was 403 at the 2010 census."," It is part of the Little Rock\u2013North Little Rock\u2013Conway Metropolitan Statistical Area."," Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century."," Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr."]],["Tyler Bates",["Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games."," Much of his work is in the action and horror film genres, with films like \"Dawn of the Dead, 300, Sucker Punch,\" and \"John Wick.\""," He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn."," With Gunn, he has scored every one of the director's films; including \"Guardians of the Galaxy\", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel."," In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums \"The Pale Emperor\" and \"Heaven Upside Down\"."]],["Ed Wood",["Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."]],["Deliver Us from Evil (2014 film)",["Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer."," The film is officially based on a 2001 non-fiction book entitled \"Beware the Night\" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was \"inspired by actual accounts\"."," The film stars Eric Bana, \u00c9dgar Ram\u00edrez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014."]],["Adam Collis",["Adam Collis is an American filmmaker and actor."," He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010."," He also studied cinema at the University of Southern California from 1991 to 1997."," Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995)."," In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\"."]],["Sinister (film)",["Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill."," It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger."]],["Conrad Brooks",["Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor."," He moved to Hollywood, California in 1948 to pursue a career in acting."," He got his start in movies appearing in Ed Wood films such as \"Plan 9 from Outer Space\", \"Glen or Glenda\", and \"Jail Bait.\""," He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor."," He also has since gone on to write, produce and direct several films."]],["Doctor Strange (2016 film)",["Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures."," It is the fourteenth film of the Marvel Cinematic Universe (MCU)."," The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton."," In \"Doctor Strange\", surgeon Strange learns the mystic arts after a career-ending car accident."]]],
        # "type":"comparison",
        # "level":"hard"},
        
        data["question"] = data["question"]
        
        # # Extract the context information
        # ctxs = data["ctxs"]
        # ctx_str = "\n".join([f"{i+1}. {ctx[1]}" for i, ctx in enumerate(ctxs)])
        
        # Extract the chain of thought (CoT)
        cot = data["context"]
        data["context"] = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot)])
        
        # Extract the answer
        data["answer"] = data["answer"]
        
        # Construct the query prompt
        query_prompt = (
            "Question: {question}. \nAnswer:"#Your answer should be 'true' or 'false'. #"f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {question}. \nAnswer:'"#Your answer should be 'true' or 'false'. 
        )
        
        # Format the query prompt with the data
        query = query_prompt.format_map(data)
        return query

    elif dataset_name == 'StrategyQA':
        
        # Extract the question
        #数据集格式
        #     {
        #     "qid": "1b29d402c3e17cb3b435",
        #     "term": "Pound sterling",
        #     "description": "Official currency of the United Kingdom and other territories",
        #     "question": "Is a pound sterling valuable?",
        #     "answer": false,
        #     "facts": [
        #         "A pound sterling is fiat money.",
        #         "Fiat money is backed by government decree and has no intrinsic value.",
        #         "One pound sterling is worth about 1.24 US dollars by May of 2020."
        #     ],
        #     "decomposition": [
        #         "What is the value of the Pound Sterling based on?",
        #         "Is #1 the material used in making it?"
        #     ],
        #     "evidence": [
        #         [
        #             [
        #                 [
        #                     "Pound sterling-16"
        #                 ]
        #             ],
        #             [
        #                 [
        #                     "Pound sterling-16"
        #                 ]
        #             ]
        #         ],
        #         [
        #             [
        #                 [
        #                     "Pound sterling-1",
        #                     "Pound sterling-12"
        #                 ]
        #             ],
        #             [
        #                 [
        #                     "Pound sterling-71"
        #                 ]
        #             ]
        #         ],
        #         [
        #             [
        #                 [
        #                     "Pound sterling-16"
        #                 ]
        #             ],
        #             [
        #                 [
        #                     "One pound (British coin)-3"
        #                 ],
        #                 "operation"
        #             ]
        #         ]
        #     ]
        # }, 
        data["question"] = data["question"]
        
        # # Extract the context information
        # ctxs = data["ctxs"]
        # ctx_str = "\n".join([f"{i+1}. {ctx[1]}" for i, ctx in enumerate(ctxs)])
        
        # Extract the chain of thought (CoT)
        cot = data["facts"]
        data["facts"] = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot)])
        
        # Extract the answer
        data["answer"] = data["answer"]
        
        # Construct the query prompt
        query_prompt = (
            "Question: {question}. Your answer should be 'true' or 'false'. \nAnswer:"#"f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {question}. Your answer should be 'true' or 'false'. \nAnswer:'"
        )
        
        # Format the query prompt with the data
        query = query_prompt.format_map(data)
        return query
        
    elif dataset_name == 'IIRC':
        # Extract the question
        #数据集格式
        #   {
        #     "pid": "p_4754",
        #     "questions": [
        #       {
        #         "answer": {
        #           "type": "span",
        #           "answer_spans": [
        #             {
        #               "start": 141,
        #               "end": 152,
        #               "text": "Switzerland",
        #               "passage": "university of geneva"
        #             }
        #           ]
        #         },
        #         "question": "In what country did Bain attend doctoral seminars of Wlad Godzich?",
        #         "question_links": [
        #           "University of Geneva"
        #         ],
        #         "qid": "q_10839",
        #         "context": [
        #           {
        #             "passage": "main",
        #             "text": "and later attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
        #             "indices": [
        #               705,
        #               790
        #             ]
        #           },
        #           {
        #             "passage": "main",
        #             "text": "He completed M. Phil at the Geneva-based IUEE (Institute for European Studies), and later attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
        #             "indices": [
        #               625,
        #               790
        #             ]
        #           },
        #           {
        #             "passage": "University of Geneva",
        #             "text": "The University of Geneva (French: Universit\u00e9 de Gen\u00e8ve) is a public research university located in Geneva, Switzerland.",
        #             "indices": [
        #               0,
        #               119
        #             ]
        #           }
        #         ]
        #       },
        #       {
        #         "answer": {
        #           "type": "span",
        #           "answer_spans": [
        #             {
        #               "start": 93,
        #               "end": 114,
        #               "text": "Province of Salamanca",
        #               "passage": "salamanca"
        #             }
        #           ]
        #         },
        #         "question": "In what Spanish province is the city located where Bain took up Hispanic Studies at a small private college?",
        #         "question_links": [
        #           "Salamanca"
        #         ],
        #         "qid": "q_10840",
        #         "context": [
        #           {
        #             "passage": "main",
        #             "text": "In 1982 he moved to Spain, and took up Hispanic Studies in a small private college in Salamanca",
        #             "indices": [
        #               196,
        #               291
        #             ]
        #           },
        #           {
        #             "passage": "Salamanca",
        #             "text": "Salamanca ( , ) is a city in western Spain that is the capital of the Province of Salamanca",
        #             "indices": [
        #               0,
        #               91
        #             ]
        #           }
        #         ]
        #       }
        #     ],
        #     "links": [
        #       {
        #         "indices": [
        #           17,
        #           23
        #         ],
        #         "target": "London"
        #       },
        #       {
        #         "indices": [
        #           34,
        #           54
        #         ],
        #         "target": "Kingston upon Thames"
        #       },
        #       {
        #         "indices": [
        #           98,
        #           105
        #         ],
        #         "target": "Liphook"
        #       },
        #       {
        #         "indices": [
        #           136,
        #           144
        #         ],
        #         "target": "Dyslexia"
        #       },
        #       {
        #         "indices": [
        #           282,
        #           291
        #         ],
        #         "target": "Salamanca"
        #       },
        #       {
        #         "indices": [
        #           324,
        #           333
        #         ],
        #         "target": "Golo Mann"
        #       },
        #       {
        #         "indices": [
        #           378,
        #           416
        #         ],
        #         "target": "Classe pr\u00e9paratoire aux grandes \u00e9coles"
        #       },
        #       {
        #         "indices": [
        #           458,
        #           469
        #         ],
        #         "target": "Jules Ferry"
        #       },
        #       {
        #         "indices": [
        #           527,
        #           547
        #         ],
        #         "target": "Les Baux-de-Provence"
        #       },
        #       {
        #         "indices": [
        #           598,
        #           623
        #         ],
        #         "target": "Paris Nanterre University"
        #       },
        #       {
        #         "indices": [
        #           749,
        #           761
        #         ],
        #         "target": "Wlad Godzich"
        #       },
        #       {
        #         "indices": [
        #           769,
        #           789
        #         ],
        #         "target": "University of Geneva"
        #       }
        #     ],
        #     "text": "Bain was born in London. He lived Kingston upon Thames attending prep school at Highfield School (Liphook, Hampshire). He suffered from Dyslexia, and made slow progress in the educational system. In 1982 he moved to Spain, and took up Hispanic Studies in a small private college in Salamanca where he met up with friends of Golo Mann. Upon return to France he qualified for the Classe pr\u00e9paratoire aux grandes \u00e9coles. He accomplished his Kh\u00e2gne in the Lyce\u00e9 Jules Ferry. The same year he discovered a new archeological area at Les Baux-de-Provence. He accomplished his BA Humanities in the radical Paris Nanterre University. He completed M. Phil at the Geneva-based IUEE (Institute for European Studies), and later attended the doctoral seminars of Wlad Godzich in the University of Geneva.\n",
        #     "title": "Thomas Bain (Orange)"
        #   },
        # for item_data in data["questions"]:
        #     # def extract_questions_and_answers(data):

        data["question"] = data["question"]
        
        # # Extract the context information
        # ctxs = data["ctxs"]
        # ctx_str = "\n".join([f"{i+1}. {ctx[1]}" for i, ctx in enumerate(ctxs)])
        
        # Extract the chain of thought (CoT)
        cot = data["context"]
        data["context"] = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot)])
        
        # Extract the answer
        data["answer"] = data["answer"]
        
        # Construct the query prompt
        query_prompt = (
            "Question: {question}. Your answer should be 'true' or 'false'. \nAnswer:"#"f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {question}. Your answer should be 'true' or 'false'. \nAnswer:'"
        )
        
        # Format the query prompt with the data
        query = query_prompt.format_map(data)
        return query