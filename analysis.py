import json
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_options(options):
    cleaned_options = {}
    for option in options:
        try:
            letters = re.findall('[A-F]', option['extract_result'])
        except:
            letters = re.findall('[A-F]', option['model_answer'])

        letters = sorted(list(set(letters)))
        cleaned_options[option['id']] = ''.join([letter.upper() for letter in letters])

    return cleaned_options

def calculate_accuracy(answers, options, flag=True):
    correct_count = 0
    total_questions = len(options)
    total_questions = 0

    static_dict = {}
    uncorrected_1 = []
    uncorrected_2 = []

    for question in answers:
        question_id = question['id']
        cleaned_answer = ''.join(filter(str.isalpha, question['answer']))
        cleaned_option = options.get(question_id, '').upper()

        if len(cleaned_answer) == 1 and len(cleaned_option)>1:
            cleaned_option = cleaned_option[0]

        if cleaned_option == '':
            pass
        elif cleaned_answer == cleaned_option:
            correct_count += 1
            total_questions += 1
            question_type = question['exam_type']
            if question_type in static_dict.keys():
                static_dict[question_type] = static_dict[question_type] + 1
            else:
                static_dict[question_type] = 1
                
        else:
            total_questions += 1
            
            if flag:
                uncorrected_1.append(question_id)
            else:
                uncorrected_2.append(question_id)

    accuracy = (correct_count / total_questions) * 100
    # print(str(correct_count) + '/' + str(total_questions))
    return accuracy

def calculate_f1_score(answers, flag=True):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_questions = len(answers)
    total_questions = 0  # This line is redundant and can be removed

    uncorrected_1 = []
    uncorrected_2 = []

    for question in answers:
        prediction = ''.join(filter(str.isalpha, question['answer']))
        label = question['answer']

        if label == 'none':
            pass
        elif prediction == label:
            true_positives += 1
            total_questions += 1
        else:
            total_questions += 1

            if flag:
                uncorrected_1.append(question["question"])
            else:
                uncorrected_2.append(question["question"])

            if prediction != 'none':
                false_positives += 1
            if label != 'none':
                false_negatives += 1

    # Calculate Precision and Recall
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    # Calculate F1 Score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def calculate_accuracy_2(answers, flag=True):
    correct_count = 0
    total_questions = len(answers)
    total_questions = 0

    static_dict = {}
    uncorrected_1 = []
    uncorrected_2 = []

    for question in answers:
        # question_id = question['id']
        prediction = str(question['extract_result']).lower()
        label = str(question['answer']).lower()

        print("label",label)
        print("prediction",prediction)
        if label == 'none':
            pass
        elif prediction == label:
            correct_count += 1
            total_questions += 1   
        else:
            total_questions += 1
            
            if flag:
                uncorrected_1.append(question["question"])
            else:
                uncorrected_2.append(question["question"])

    accuracy = (correct_count / total_questions) * 100
    # print(str(correct_count) + '/' + str(total_questions))
    return accuracy


def evaluate(options_file_path):
    #CMB-Exam        EM
    #MMCU-Medical    EM
    #CMB-Clin        BLEU-1 BLEU-4 ROUGE
    #2WikiMultihopQA EM F1
    #HotpotQA        EM F1
    #StrategyQA      Accuracy
    #IIRC           EM F1
    
    # 提取result_dir
    datasets=['CMB', 'MMCU', 'Clin','2WikiMultihopQA','HotpotQA', 'StrategyQA','IIRC']
    dataset_name='CMB'
    for dataset in datasets:
        if dataset.lower() in options_file_path.lower():
            dataset_name=dataset

    options = load_json(options_file_path)

    
    if dataset_name in ['CMB','MMCU','Clin']:
        cleaned_options = clean_options(options)

        accuracy = calculate_accuracy(options, cleaned_options, False)
        
        return round(accuracy, 2)
    elif dataset_name in ['2WikiMultihopQA','HotpotQA','IIRC']:

        # cleaned_options = clean_options(options)
        F1=calculate_f1_score(options, False)
        accuracy = calculate_accuracy_2(options, False)
        print(round(accuracy,2))
        print(round(F1,2))
        return round(accuracy, 2)
    elif dataset_name in ['StrategyQA']:

        # cleaned_options = clean_options(options)
        accuracy = calculate_accuracy_2(options, False)
        print(round(accuracy,2))
        return round(accuracy, 2)

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    if args.dataset_name != 'Clin':
        accuracy = evaluate(args.ex_result_path)
        print(f"Accurary: {accuracy}%")
    else:
        from clinical import evaluate_clinical
        evaluate_clinical(args)