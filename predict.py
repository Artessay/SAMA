import json
import datetime

from data import load_dataset, get_query
from model import load_agent
from utils import seed_everything

def run(args):
    seed_everything(args.seed)

    checkpoint = args.checkpoint
    checkpoint_dir = args.checkpoint_dir

    res_list = []
    if checkpoint != 0:
        f = open('{}/test_before_{}.json'.format(checkpoint_dir, checkpoint), 'r')
        content = f.read()
        res_list = json.loads(content)
        f.close()

    agent = load_agent(args.module_name, args)
    data_list = load_dataset(args.dataset_name)

    for id, item in enumerate(data_list):
        if id + 1 < checkpoint:
            continue

        data = get_query(item,args.dataset_name)
        
        print("{}/{}".format(id+1, len(data_list)))

        if (id+1) % 10 == 0:
            with open('{}/test_before_{}.json'.format(checkpoint_dir, id+1), 'w') as f:
                json.dump(res_list, f, indent=4, ensure_ascii=False)

        starttime = datetime.datetime.now()
        response = agent.inference(data)
        endtime = datetime.datetime.now()

        item["result"] = response

        res_list.append(item)

        print("LLM answer:", response)
        print("Gold answer:", item["answer"])
        print("use time:", (endtime - starttime).seconds, "s")
        print()

    with open(args.result_path, 'w') as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    from utils import get_args
    args = get_args()
    run(args)