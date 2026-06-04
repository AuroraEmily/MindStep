import json
import argparse
from openai import OpenAI
import openai
import os
import csv
from tqdm import tqdm
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="intent_rec.json")
    parser.add_argument("--model", type=str,
                        choices=['deepseek-v4-flash', 'gpt-5-mini', "qwen3-max-2026-01-23",'gpt-4o-mini'])
    parser.add_argument("--cot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--mindstep", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Use structured cognitive evocation prompt")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process. -1 for all.")
    args = parser.parse_args()
    return args


def get_letters_and_range(dataset_type):
    """letter range"""
    if "intent_rec" in dataset_type:
        return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], "A-J"
    elif "intent_seeker" in dataset_type:
        return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"], "A-P"
    else:
        return ["A", "B", "C", "D", "E"], "A-E"


# ========== MindStep prompts ==========


seeker_mindstep = '''To infer the seeker's intention, simulate the cognitive processing step by step. Please follow this format strictly, No extra words:
[Evocation-1] Step into the seeker's shoes. Walk through the dialogue history to capture the subtle context and feel how the conversational vibe has evolved.
[Percept-2] Spot the target utterance. Catch your immediate reaction, reading between the lines to grasp its literal meaning, hidden implications, and shifts in intent.
[Reaction-3] From your view, sense the drive: hunting for recs, reacting to a pick, or just chatting? If reacting, are you accepting, rejecting, or critiquing?
Therefore the answer is: 'XX' (where X is a combination of letters, e.g., 'A', 'BC', 'ADE')'''




rec_mindstep = '''To infer the recommender's intention, simulate the cognitive processing step by step. Please follow this format strictly, No extra words:
[Evocation-1] Step into the recommender's shoes. Summarize the seeker's current state and implicit needs based on the prior dialogue context.
[Percept-2] Focus strictly on the seeker's latest input. Identify the exact query, feedback, or emotional cue that serves as the direct trigger for your next response.
[Reaction-3] Analyze the target utterance. Pinpoint its primary conversational goal (e.g., asking for preferences, recommending, chitchatting, greeting, or giving feedback,...) and map your motivation directly to the provided choices.
Therefore the answer is: X (where X is a combination of letters, e.g., 'A', 'BC', 'ADE')'''



def extract_answers(response, answer_range):
    response = re.sub(r'["\']', '', response)
    response = re.sub(r'\b(?:and|or|&|,|;)\b', ' ', response, flags=re.IGNORECASE)
    response = re.sub(r'[^A-Z\s]', '', response.upper())
    A, Z = answer_range.split('-')
    valid_letters = ''.join(chr(c) for c in range(ord(A.upper()), ord(Z.upper()) + 1))

    lead_pattern = r"\b(?:answer\s*(?:is|:)|the\s+answer\s+is)\b"

    answer_block_pattern = r"[^{}]*((?:[{}]|\\boxed\{{[{}]}})+)".format(
        valid_letters, valid_letters, valid_letters
    )

    full_pattern = r"(?i){}{}".format(lead_pattern, answer_block_pattern)

    matches = re.findall(full_pattern, response)

    all_letters = []

    for match in matches:
        answer_block = match

        letters = re.findall(
            r"\\boxed\{{([{}])}}|([{}])".format(valid_letters, valid_letters),
            answer_block,
            re.IGNORECASE
        )

        letters = [x[0].upper() or x[1].upper() for x in letters]
        all_letters.extend(letters)

    unique_letters = sorted(set(all_letters))

    return unique_letters if unique_letters else None


def evaluate(args, client, problem, idx):
    answers = []
    letters, answer_range = get_letters_and_range(args.dataset_type)

    if args.MindStep:
        # MindStep
        if "intent_rec" in args.dataset_type:
            system_prompt = rec_mindstep
        elif "intent_seeker" in args.dataset_type:
            system_prompt = seeker_mindstep

        else:
            print('error: unknown dataset type for MindStep')
            system_prompt = ""

    elif args.cot:
        system_prompt = """Here is a movie recommendation dialogue, there are two agents, the RECOMMENDER and the SEEKER. The RECOMMENDER is trying to recommend movies to SEEKER. Think step by step to answer the question, but limit yourself to no more than 3 steps."""
    else:
        system_prompt = """You are an expert in dialogue analysis. Given a dialogue and a multiple-choice question, respond ONLY with the letter(s) of the correct choice(s) from A-P. Do not include any other text, punctuation, explanation, or whitespace. Example valid outputs: 'A', 'BC', 'ADE'."""

    utterance_context = problem["utterance_context"]
    question = problem["question"]


    if "choices" in problem:
        choices = problem["choices"]
    elif "choice" in problem:
        choices = problem["choice"]
    else:
        choices = []
    choices_str = "\n".join([f"{choice}" for choice in choices])

    if args.MindStep:
        user_prompt = "\nDialogue History:\n" + utterance_context + "\nQuestion:\n" + question + "\nChoices:\n" + choices_str + "\n"
    elif args.cot:
        shot = """\nEnding with "The answer is X", where X is a combination of letters from choices (e.g., AB, ACD).
Do not use any other format for the ending.
Multiple selections are valid and expected when appropriate."""
        user_prompt = shot + "\nDialogue History:\n" + utterance_context + "\nQuestion:\n" + question + "\nChoices:\n" + choices_str + "\nAnswer: Let's think step by step."
        print(user_prompt)
    else:
        user_prompt = "\nDialogue History:\n" + utterance_context + "\nQuestion:\n" + question + "\nChoices:\n" + choices_str + "\nAnswer:"
        print(user_prompt)

    if "o1" in args.model or "gemma" in args.model:
        messages = [{"role": "user", "content": system_prompt + "\n" + user_prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    while True:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=3000,

        )
        # when use 'deepseek-v4-flash',you need set thinking disabled,it's default is enabled
        # response = client.chat.completions.create(
        #     model=args.model,
        #     messages=messages,
        #     temperature=0.1,
        #     extra_body={"thinking": {"type": "disabled"}},
        #     max_completion_tokens=3000,
        #
        # )
        result = response.choices[0].message.content
        print("\n", response.choices[0].message)

        if args.cot or args.MindStep:
            try:
                candidates = extract_answers(result, answer_range)

                if not candidates:
                    candidates = ["Y"]
                    print("!!!\n", response.choices[0].message)
                    break
            except:
                candidates = ["Z"]
                break
        else:
            cleaned = re.sub(r'[^A-Za-z]', '', result)
            candidates = list(set(c for c in cleaned))

        if all(i in letters for i in candidates):
            break

    dialogue_id = problem["dialogue_id"]
    utterance = problem["utterance_pos"]
    candidates = list(set(candidates))
    if "intent_rec" in args.dataset_type:
        key = problem["answer_fine"]
    elif "intent_seeker" in args.dataset_type:
        key = problem["answer_fine"]


    answers = [dialogue_id, utterance, key, candidates]
    return answers, idx


if __name__ == '__main__':
    args = arg_parse()
    with open(f"../data/RecTom/{args.dataset_type}", 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.num_samples > 0:
        data = data[:args.num_samples]
        print(f"--- Running on a subset of {len(data)} samples ---")
    if 'deepseek-v4-flash' in args.model:
        API_BASE = ""
        API_KEY = ""
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    elif 'gpt-4o-mini' in args.model:
        API_BASE = ""
        API_KEY = ""
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    elif 'gpt-5-mini' in args.model:
        API_BASE = ""
        API_KEY = ""
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    elif 'qwen3-max-2026-01-23' in args.model:
        API_BASE = ""
        API_KEY = ""
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.MindStep:
        file_path = f'..outputs//RecTom/{args.model}/{args.dataset_type}_mindstep_{timestamp}.csv'
    elif args.cot:
        file_path = f'..outputs//RecTom/{args.model}/{args.dataset_type}_cot_{timestamp}.csv'
    else:
        file_path = f"..outputs//RecTom/{args.model}/{args.dataset_type}_{timestamp}.csv"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    all_results = [None] * len(data)

    correct_predictions = [0]
    completed_count = [0]
    lock = threading.Lock()


    def update_accuracy_and_print(idx):
        with lock:
            current_result = all_results[idx]
            if sorted(current_result[2]) == sorted(current_result[3]):
                correct_predictions[0] += 1
            completed_count[0] += 1
            current_accuracy = correct_predictions[0] / completed_count[0]
            print(
                f"Task {completed_count[0]}/{len(data)}: Current accuracy is {current_accuracy:.4f} ({correct_predictions[0]}/{completed_count[0]})")


    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(evaluate, args, client, problem, idx): idx
            for idx, problem in enumerate(tqdm(data, desc="submitting tasks"))
        }
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dialogue_id', 'utterance', 'labels', 'predictions'])

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
            try:
                result, idx = future.result()
                all_results[idx] = result
                update_accuracy_and_print(idx)
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if result is not None:
                        writer.writerow(result)
            except Exception as exc:
                print(f"Task generated an exception: {exc}")


        if completed_count[0] > 0:
            final_accuracy = correct_predictions[0] / completed_count[0]
            print(f"\n====================================")
            print(f"Task Finished! Total completed: {completed_count[0]}/{len(data)}")
            print(f"Final Accuracy: {final_accuracy:.4f} ({correct_predictions[0]}/{completed_count[0]})")
            print(f"Results saved to: {file_path}")
            print(f"====================================")
