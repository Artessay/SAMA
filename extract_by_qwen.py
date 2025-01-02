import json
import time
import datetime

from microservice import CustomLanguageModel


def extract_answer(args):
    if args.dataset_name in ['CMB','MMCU','Clin']:
        llm = CustomLanguageModel("Aliyun", use_local=False)
        with open(args.result_path, 'r') as f:
            data_list = json.load(f)

        res_list = []
        checkpoint = args.checkpoint
        checkpoint_dir = args.checkpoint_dir
        if checkpoint != 0:
            f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
            content = f.read()
            res_list = json.loads(content)
            f.close()

        print(len(data_list))
        for item in data_list:
            question =item["option_str"]
            id = item["id"]
            if id < checkpoint:
                continue

            print("{}/{}".format(id, len(data_list)))
            data = "下面有一些选项和学生的回答，我需要你理解学生的回答，告诉我学生的答案具体哪个或哪些选项\n"+"【选项】\n"+question
            data = data + "\n【学生回答】\n"+item["result"]+"\n请直接回答学生回答的是问题里的哪个或哪些选项(只回答选项的字母，不要回答选项的内容)，不要回答额外的内容。请回答选择的是选项:"
            starttime = datetime.datetime.now()

            if id % 10 == 0:
                json.dump(res_list, open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, id), 'w'),
                        indent=4, ensure_ascii=False)
                
            for i in range(10):
                try:
                    res = llm(data)
                    break
                except Exception as e:
                    print(e)
                    if "Input data may contain inappropriate content" in str(e):
                        res = 'X'
                        break
                    if i == 4:
                        exit(0)
                    time.sleep(10)

            item["extract_result"] = res
            res_list.append(item)
            endtime = datetime.datetime.now()

            print("Gold answer:", item["answer"])
            print("Pred answer:", res)
            print("use time:", (endtime - starttime).seconds, "s")

        with open(args.ex_result_path, 'w') as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False)
            
    elif args.dataset_name == '2WikiMultihopQA':
        #    {
        #     "_id": "61a46987092f11ebbdaeac1f6bf848b6",
        #     "type": "comparison",
        #     "question": "Which film came out first, Blind Shaft or The Mask Of Fu Manchu?",
        #     "context": "1. ['The Return of Dr. Fu Manchu', ['The Return of Dr. Fu Manchu is a 1930 American pre-Code film directed by Rowland V. Lee.', 'It is the second of three films starring Warner Oland as the fiendish Fu Manchu, who returns from apparent death in the previous film,\" The Mysterious Dr. Fu Manchu\"( 1929), to seek revenge on those he holds responsible for the death of his wife and child.']]\n2. ['The Blood of Fu Manchu', ['The Blood of Fu Manchu, also known as Fu Manchu and the Kiss of Death, Kiss of Death, Kiss and Kill( U.S. title) and Against All Odds( original U.S. video title), is a 1968 British adventure crime film directed by Jesús Franco, based on the fictional Asian villain Dr. Fu Manchu created by Sax Rohmer.', 'It was the fourth film in a series, and was preceded by\" The Vengeance of Fu Manchu The Castle of Fu Manchu\" followed in 1969.', 'It was produced by Harry Alan Towers for Udastex Films.', 'It starred Christopher Lee as Dr. Fu Manchu, Richard Greene as Scotland Yard detective Nayland Smith, and Howard Marion- Crawford as Dr. Petrie.', 'The movie was filmed in Spain and Brazil.', 'Shirley Eaton appears in a scene that she claimed she was never paid for; apparently, the director Jesús Franco had inserted some stock footage of her from one of her films(\" The Girl from Rio\"( 1968)) into the film without telling her.', 'She only found out years later that she had been in a Fu Manchu film.']]\n3. ['The Mask of the Gorilla', ['The Mask of the Gorilla is a 1958 French action film directed by Bernard Borderie.']]\n4. ['The Mask of Fu Manchu', ['The Mask of Fu Manchu is a 1932 pre-Code adventure film directed by Charles Brabin.', 'It was written by Irene Kuhn, Edgar Allan Woolf and John Willard based on the 1932 novel of the same name by Sax Rohmer.', \"Starring Boris Karloff as Fu Manchu, and featuring Myrna Loy as his depraved daughter, the movie revolves around Fu Manchu's quest for the golden sword and mask of Genghis Khan.\", 'Lewis Stone plays his nemesis.', 'Dr. Petrie is absent from this film.']]\n5. ['The Face of Fu Manchu', ['The Face of Fu Manchu is a 1965 thriller film directed by Don Sharp and based on the characters created by Sax Rohmer.', 'It stars Christopher Lee as the eponymous villain, a Chinese criminal mastermind, and Nigel Green as his pursuing rival Nayland Smith, a Scotland Yard detective.', 'The film was a British- West German co-production, and was the first in a five- part series starring Lee and produced by Harry Alan Towers for Constantin Film, the second of which was\" The Brides of Fu Manchu\" released the next year, with the final entry being\" The Castle of Fu Manchu\" in 1969.', 'It was shot in Technicolor and Techniscope, on- location in County Dublin, Ireland.']]\n6. ['The Brides of Fu Manchu', ['The Brides of Fu Manchu is a 1966 British/ West German Constantin Film co-production adventure crime film based on the fictional Chinese villain Dr. Fu Manchu, created by Sax Rohmer.', 'It was the second film in a series, and was preceded by\" The Face of Fu ManchuThe Vengeance of Fu Manchu\" followed in 1967,\" The Blood of Fu Manchu\" in 1968, and\" The Castle of Fu Manchu\" in 1969.', 'It was produced by Harry Alan Towers for Hallam Productions.', 'Like the first film, it was directed by Don Sharp, and starred Christopher Lee as Fu Manchu.', 'Nigel Green was replaced by Douglas Wilmer as Scotland Yard detective Nayland Smith.', 'The action takes place mainly in London, where much of the location filming took place.']]\n7. ['The Castle of Fu Manchu', ['The Castle of Fu Manchu( also known as The Torture Chamber of Dr. Fu Manchu and also known by its German title Die Folterkammer des Dr. Fu Man Chu) is a 1969 film and the fifth and final Dr. Fu Manchu film with Christopher Lee portraying the title character.']]\n8. ['The Mysterious Dr. Fu Manchu', ['The Mysterious Dr. Fu Manchu is a 1929 American pre-Code drama film directed by Rowland V. Lee and starring Warner Oland as Dr. Fu Manchu.', 'It was the first Fu Manchu film of the talkie era.', 'Since this was during the transition period to sound, a silent version was also released in the United States.']]\n9. ['Les Trottoirs de Bangkok', ['Les Trottoirs de Bangkok( English:\" The Sidewalks of Bangkok\", also known as\" Bangkok Interdit\") is a 1984 erotic thriller film directed by Jean Rollin.', 'The film was inspired by the 1932 Boris Karloff classic\" The Mask of Fu Manchu\".', 'In contrast to Rollin\\'s usual themes of vampires, dream- like atmosphere and crumbling châteaus,\" Les Trottoirs des Bangkok\" mixes themes of adventure, crime and mystery with comic book dialogue, while still featuring naked women and sex, which his films were known for.']]\n10. ['Blind Shaft', ['Blind Shaft is a 2003 film about a pair of brutal con artists operating in the illegal coal mines of present- day northern China.', 'The film was written and directed by Li Yang( 李杨), and is based on Chinese writer Liu Qingbang\\'s short novel\" Shen MuSacred Wood\").']]",
        #     "entity_ids": "Q851284_Q1214768",
        #     "supporting_facts": [
        #         [
        #             "Blind Shaft",
        #             0
        #         ],
        #         [
        #             "The Mask of Fu Manchu",
        #             0
        #         ]
        #     ],
        #     "evidences": [
        #         [
        #             "Blind Shaft",
        #             "publication date",
        #             "2003"
        #         ],
        #         [
        #             "The Mask of Fu Manchu",
        #             "publication date",
        #             "1932"
        #         ]
        #     ],
        #     "answer": "The Mask Of Fu Manchu",
        #     "evidences_id": [],
        #     "answer_id": "Q1214768",
        #     "result": "Final Answer:电影《The Mask of Fu Manchu》比《Blind Shaft》早。《The Mask of Fu Manchu》于1967年上映，而《Blind Shaft》则于1980年上映。"
        # },


        with open(args.result_path, 'r') as f:
            data_list = json.load(f)

        res_list = []
        checkpoint = args.checkpoint
        checkpoint_dir = args.checkpoint_dir
        if checkpoint != 0:
            f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
            content = f.read()
            res_list = json.loads(content)
            f.close()

        # print(len(data_list))
        for item in data_list:
            answer= item["result"]
            if answer.lower().startswith("final answer:"):
                # 去除前缀
                answer=answer[len("final answer:"):].strip()
            else:
                # 如果前缀不存在，直接返回原字符串
                answer=answer
            item["extract_result"] = answer
            
            
            res_list.append(item)
            # print("Gold answer:", item["answer"])
            # print("Pred answer:", item["result"])
        with open(args.ex_result_path, 'w') as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False)        
        
        
        
    elif args.dataset_name == 'HotpotQA':
        with open(args.result_path, 'r') as f:
            data_list = json.load(f)

        res_list = []
        checkpoint = args.checkpoint
        checkpoint_dir = args.checkpoint_dir
        if checkpoint != 0:
            f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
            content = f.read()
            res_list = json.loads(content)
            f.close()

        # print(len(data_list))
        for item in data_list:
            answer= item["result"]
            if answer.lower().startswith("final answer:"):
                # 去除前缀
                answer=answer[len("final answer:"):].strip()
            else:
                # 如果前缀不存在，直接返回原字符串
                answer=answer
            item["extract_result"] = answer
            res_list.append(item)
            # print("Gold answer:", item["answer"])
            # print("Pred answer:", item["result"])
        with open(args.ex_result_path, 'w') as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False)       

            
    elif args.dataset_name == 'StrategyQA':
        with open(args.result_path, 'r') as f:
            data_list = json.load(f)

        res_list = []
        checkpoint = args.checkpoint
        checkpoint_dir = args.checkpoint_dir
        if checkpoint != 0:
            f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
            content = f.read()
            res_list = json.loads(content)
            f.close()

        # print(len(data_list))
        for item in data_list:
            #这个地方要进行修改
            if ('true' in item["result"].lower()) and (not('false' in item["result"].lower()) ):
                item["extract_result"] ='true'
            elif  (not('true' in item["result"].lower())) and ('false' in item["result"].lower() ):
                item["extract_result"] ='false'
            else:
                item["extract_result"]=item["result"]
            res_list.append(item)
            # print("Gold answer:", item["answer"])
            # print("Pred answer:", item["result"])
        with open(args.ex_result_path, 'w') as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False) 
            
    elif args.dataset_name == 'IIRC':
        
        with open(args.result_path, 'r') as f:
            data_list = json.load(f)

        res_list = []
        checkpoint = args.checkpoint
        checkpoint_dir = args.checkpoint_dir
        if checkpoint != 0:
            f = open('{}/ex_test_full_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
            content = f.read()
            res_list = json.loads(content)
            f.close()

        # print(len(data_list))
        for item in data_list:
            #这个地方要进行修改
            #none "none"
            #span 啥都有
            #value value+unit
            #binary  "yes" "no"
            #暂时不分类处理了

            answer= item["result"]
            if answer.lower().startswith("final answer:"):
                # 去除前缀
                answer=answer[len("final answer:"):].strip()
            else:
                # 如果前缀不存在，直接返回原字符串
                answer=answer
            item["extract_result"] = answer
            res_list.append(item)
            # print("Gold answer:", item["answer"])
            # print("Pred answer:", item["result"])
        with open(args.ex_result_path, 'w') as f:
            json.dump(res_list, f, indent=4, ensure_ascii=False) 

if __name__ == '__main__':
    from utils import get_args
    from dotenv import load_dotenv
    load_dotenv()

    args = get_args()
    extract_answer(args)