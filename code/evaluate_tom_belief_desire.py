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
    parser.add_argument("--dataset_type", type=str, default="desire_seeker_com.json")
    parser.add_argument("--model", type=str,
                        choices=['deepseek-v4-flash','gpt-5-mini',"qwen3-max-2026-01-23", 'gpt-4o-mini'])
    parser.add_argument("--cot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--mindstep", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process. -1 for all.")
    args = parser.parse_args()
    return args


letters = ["A", "B", "C", "D", "E", "F", "G"]

# ================= mindstep prompts =================

desire_mindstep = '''To predict the seeker's most likely action, simulate the cognitive processing step by step. Please follow this format strictly, No extra words:
[Evocation-1] Summarize seeker's preferred genres and explicitly rejected types.
[Percept-2] Locate the target movie. Extract seeker's exact reaction (accepted, liked, seen-it, or rejected).
[Reaction-3] Determine likelihood: "Likely to watch" includes wanting to watch, OR having watched and liked it. Only choose "no" if explicitly rejected or disliked.
Therefore the answer is: X (where X is A or B)'''


belief_mindstep = '''To infer the recommender's belief, simulate the cognitive processing step by step. Please follow this format strictly, No extra words:
[Evocation-1] Step into the recommender's shoes. Recall your interaction and get a feel for the seeker's general taste.
[Percept-2] Spot the target movie. Notice who brought it up. Did the seeker watch it? Mentioning plot details or saying "seen it" means yes.
[Reaction-3] From the recommender's view, sense the seeker's attitude: are they positive (accepting, eager, or fond if seen) or resistant (declining or critical if seen)?
Therefore the answer is: X (where X is one of A-G)'''



def extract_answers(response, answer_range):
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

    if "desire" in args.dataset_type:
        answer_range = 'A-B'
    elif "belief" in args.dataset_type:
        answer_range = 'A-G'
    else:
        answer_range = 'A-G'

    if args.MindStep:
        # MindStep
        if "desire" in args.dataset_type:
            system_prompt = desire_mindstep
        elif "belief" in args.dataset_type:
            system_prompt = belief_mindstep

        else:
            print('erro')

        system_prompt=system_prompt
        user_prompt = "\nDialogue History:\n" + problem["utterance_context"] + "\nQuestion:\n" + problem[
            "question"] + "\nChoices:\n" + "\n".join(
            [f"{key}: {value}" for key, value in problem["choices"].items()])

    elif args.cot:
        # 2.  Zero-shot CoT
        system_prompt = """Here is a movie recommendation dialogue, there are two agents, the RECOMMENDER and the SEEKER. The RECOMMENDER is trying to recommend movies to SEEKER. Think step by step to answer the quesiton, but limit yourself to no more than 3 steps."""
        shot = """\nEnding with "The answer is X", where X is one of the option from choices.\nDo not use any other format for the ending."""
        user_prompt = shot + "\nDiallogue History:\n" + problem["utterance_context"] + "\nQuestion:\n" + problem[
            "question"] + "\nChoices:\n" + "\n".join(
            [f"{key}: {value}" for key, value in problem["choices"].items()]) + "\nAnswer: Let's think step by step."

    else:
        # 3. Zero-shot
        system_prompt = """You are an expert in dialogue analysis. Given a dialogue and a question, respond ONLY with the letter of the correct choice from A-G. Do not include any other text, punctuation, explanation, or whitespace. Example valid outputs: 'A', 'D'."""
        user_prompt = "\nDiallogue History:\n" + problem["utterance_context"] + "\nQuestion:\n" + problem[
            "question"] + "\nChoices:\n" + "\n".join(
            [f"{key}: {value}" for key, value in problem["choices"].items()]) + "\nAnswer:"
    # =========================================================

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
        #when use 'deepseek-v4-flash',you need set thinking disabled,it's default is enabled
        # response = client.chat.completions.create(
        #     model=args.model,
        #     messages=messages,
        #     temperature=0.1,
        #     extra_body={"thinking": {"type": "disabled"}},
        #     max_completion_tokens=3000,
        #
        # )
        result = response.choices[0].message.content
        # ====== print raw outputs ======
        # print(f"\n=== [Task {idx}] Model Raw Output ===")
        # print(result)
        # print("===================================")
        if not args.mindstep and not args.cot:
            cleaned = re.sub(r'[^A-Za-z]', '', result)
            candidates = list(set(c for c in cleaned))
        else:
            try:

                candidates = extract_answers(result, answer_range)
                if not candidates:
                    candidates = ["Y"]
                    print("!!!\n", response.choices[0].message)

                    break
            except:
                candidates = ["Z"]

                break
            if all(i in letters for i in candidates):
                break

        if all(i in letters for i in candidates):
            break

    dialogue_id = problem["dialogue_id"]
    utterance = problem["utterance_pos"]
    candidates = list(set(candidates))
    answers = [dialogue_id, utterance, problem["answer"], candidates]
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
        # ====== accuracy ======
        if len(data) > 0:
            final_accuracy = correct_predictions[0] / completed_count[0]
            print(f"\n====================================")
            print(f"Task Finished! Total completed: {completed_count[0]}/{len(data)}")
            print(f"Final Accuracy: {final_accuracy:.4f} ({correct_predictions[0]}/{completed_count[0]})")
            print(f"Results saved to: {file_path}")
            print(f"====================================")
