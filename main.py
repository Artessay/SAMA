from predict import run
from utils import get_args
from analysis import evaluate
from dotenv import load_dotenv
from extract_by_qwen import extract_answer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,5,6,7"

def main():
    load_dotenv()
    args = get_args()

    run(args)
    extract_answer(args)
    evaluate(args.ex_result_path)

if __name__ == "__main__":
    main()