from predict import run
from utils import get_args
from analysis import evaluate
from dotenv import load_dotenv
from extract_by_qwen import extract_answer
import os

def main():
    load_dotenv()
    args = get_args()

    run(args)
    extract_answer(args)
    evaluate(args.ex_result_path)

if __name__ == "__main__":
    main()