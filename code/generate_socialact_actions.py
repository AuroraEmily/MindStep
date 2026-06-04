# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from prompt.prompts import (
    MindStep, COT, Direct_output,
    MindStep_reverse, MindStep_noevo, MindStep_noper
)


# ====================arg ====================
def arg_parse():
    parser = argparse.ArgumentParser(description="SocialAct model inference script")
    parser.add_argument("--model", type=str, required=True,
                        choices=['deepseek-v4-flash', 'gpt-5-mini', 'qwen3-max-2026-01-23', 'gpt-4o-mini'],
                        help="Model to use")
    parser.add_argument("--event", type=str, required=True,
                        help="Event name, e.g., BlackLivesMatter, Covid, MeToo, Wildfire")
    parser.add_argument("--method", type=str, required=True,
                        choices=['MindStep', 'MindStep_reverse', 'MindStep_noevo',
                                 'MindStep_noper', 'COT', 'Direct_output'],
                        help="Prompting method")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    parser.add_argument("--data_root", type=str,
                        default=r"..\data\SocialAct",
                        help="Root directory containing raw data (_Question.json and _1000.csv)")
    parser.add_argument("--output_root", type=str,
                        default=r"..\outputs\SocialAct",
                        help="Root output directory (model subdirectory will be created under it)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing output file (skip already processed samples)")
    return parser.parse_args()


# ==================== model ====================
def create_client(model_name):
    """ OpenAI client"""
    if 'deepseek-v4-flash' in model_name:
        API_BASE = ""
        API_KEY = ""
    elif 'gpt-5-mini' in model_name:
        API_BASE = ""
        API_KEY = ""
    elif 'qwen3-max-2026-01-23' in model_name:
        API_BASE = ""
        API_KEY = ""

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return OpenAI(api_key=API_KEY, base_url=API_BASE)


def call_model(client, model_name, input_text, max_retries=3, retry_delay=5):
    """response"""
    messages = [{"role": "user", "content": input_text}]
    for attempt in range(max_retries):
        try:
            # Qwen and DeepSeek should set thinking false
            if 'qwen' in model_name or 'deepseek' in model_name:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    extra_body={"enable_thinking": False},
                    stream=False,

                )
            else:
                # GPT
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    stream=False
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


# ==================== resume ====================
def load_existing_results(output_json_path):
    """load file"""
    if not os.path.exists(output_json_path):
        return {}, []
    try:
        with open(output_json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}, []
        existing_dict = {}
        ordered_list = []
        for item in data:
            if isinstance(item, dict) and len(item) == 1:
                tid = list(item.keys())[0]
                existing_dict[tid] = item
                ordered_list.append(item)
        return existing_dict, ordered_list
    except Exception as e:
        print(f"Warning: Could not load existing output file: {e}")
        return {}, []


def save_results(output_json_path, results_list):
    """save result JSON """
    with open(output_json_path, 'w', encoding='utf-8-sig') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)


# ==================== main ====================
def main():
    args = arg_parse()

    # 1. input_path
    question_json = os.path.join(args.data_root, f"{args.event}_Question.json")
    csv_path = os.path.join(args.data_root, f"{args.event}_1000.csv")
    if not os.path.exists(question_json):
        raise FileNotFoundError(f"Question file not found: {question_json}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 2. raw question
    with open(question_json, 'r', encoding='utf-8-sig') as f:
        question_dict = json.load(f)   # {tweet_id: original_text}

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if args.num_samples > 0:
        df = df.head(args.num_samples)
        print(f"Limiting to first {args.num_samples} samples")

    # 3. add suffix
    suffix_map = {
        'MindStep': MindStep,
        'COT': COT,
        'Direct_output': Direct_output,
        'MindStep_reverse': MindStep_reverse,
        'MindStep_noevo': MindStep_noevo,
        'MindStep_noper': MindStep_noper,
    }
    suffix = suffix_map.get(args.method)
    if suffix is None:
        raise ValueError(f"Unknown method: {args.method}")

    # 4. client
    client = create_client(args.model)

    # 5. output path
    output_dir = os.path.join(args.output_root, args.model)
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{args.event}_{args.method}.json")

    # 6. load results
    existing_dict, ordered_results = load_existing_results(output_json_path)
    processed_ids = set(existing_dict.keys())
    if args.resume:
        print(f"Resume mode: found {len(processed_ids)} previously processed samples.")
    else:
        # clean
        existing_dict = {}
        ordered_results = []
        processed_ids = set()
        print("Resume disabled: starting from scratch.")

    # 7. csv
    total = len(df)
    new_count = total - len(processed_ids)
    if new_count == 0:
        print("All samples already processed. Nothing to do.")
        return
    print(f"Need to process {new_count} new samples out of {total}.")

    # use tqdm
    with tqdm(total=total, initial=len(processed_ids), desc=f"Processing {args.event}/{args.method}") as pbar:
        for idx, row in df.iterrows():
            tweet_id = str(row['id'])

            # skip processed items
            if tweet_id in processed_ids:
                pbar.update(1)
                continue


            original_text = question_dict.get(tweet_id, "")
            if not original_text:
                print(f"\n  Warning: No original text for tweet_id {tweet_id}")

            prompt = original_text + suffix


            try:
                model_output = call_model(client, args.model, prompt)
            except Exception as e:
                print(f"\n  Sample {tweet_id} finally failed: {e}")
                model_output = ""


            new_item = {tweet_id: {"input": prompt, "output": model_output}}
            ordered_results.append(new_item)
            processed_ids.add(tweet_id)


            save_results(output_json_path, ordered_results)


            pbar.update(1)


    print(f"\nDone! Output file: {output_json_path}")
    print(f"Total records in output: {len(ordered_results)}")
    missing = sum(1 for tid in processed_ids if not question_dict.get(tid, ""))
    print(f"Missing original text count: {missing}")


if __name__ == "__main__":
    main()