# -*- coding: utf-8 -*-
import os
import json
import csv
import re
import ast
import warnings
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import html
from difflib import SequenceMatcher

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==================== def ====================



def extract_action_from_output(output_text):
    if not output_text or not isinstance(output_text, str):
        return None, None, None
    lines = output_text.splitlines()
    n = len(lines)
    result = {'Action object': None, 'Action type': None, 'Action content': None}
    i = 0
    while i < n:
        line = lines[i].strip()
        line_lower = line.lower()
        if line_lower.startswith('action object:') or line_lower.startswith('action type:') or line_lower.startswith('action content:'):
            parts = line.split(':', 1)
            key_raw = parts[0].strip().lower()
            value_part = parts[1].strip() if len(parts) > 1 else ""
            if 'object' in key_raw:
                key = 'Action object'
            elif 'type' in key_raw:
                key = 'Action type'
            elif 'content' in key_raw:
                key = 'Action content'
            else:
                key = key_raw
            if value_part:
                result[key] = value_part
                i += 1
            else:
                i += 1
                value_lines = []
                while i < n:
                    next_line = lines[i].strip()
                    next_line_lower = next_line.lower()
                    if (next_line_lower.startswith('action object:') or
                        next_line_lower.startswith('action type:') or
                        next_line_lower.startswith('action content:')):
                        break
                    if next_line:
                        value_lines.append(next_line)
                    i += 1
                result[key] = ' '.join(value_lines) if value_lines else ''
                continue
        else:
            i += 1
    obj = result.get('Action object')
    typ = result.get('Action type')
    cont = result.get('Action content')
    if obj is not None and typ is not None and cont is not None:
        return obj, typ, cont
    return None, None, None


def load_ground_truth(csv_path):
    truth = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required_cols = ['id', 'action_object_answer', 'action_type_answer', 'action_content_answer', 'content_option']
        for col in required_cols:
            if col not in reader.fieldnames:
                raise KeyError(f"CSV missing '{col}' column, columns: {reader.fieldnames}")
        for row in reader:
            id_val = row['id'].strip()
            obj_ans = row['action_object_answer'].strip()
            type_ans = row['action_type_answer'].strip()
            content_ans = row['action_content_answer'].strip()
            opt_str = row['content_option'].strip()
            try:
                opt_list = ast.literal_eval(opt_str)
                if len(opt_list) != 4:
                    print(f"Warning: {id_val} content_option length not 4: {opt_list}")
            except Exception as e:
                print(f"Warning: {id_val} content_option parse failed: {opt_str}, error: {e}")
                continue
            truth[id_val] = {
                'action_object_answer': obj_ans,
                'action_type_answer': type_ans,
                'action_content_answer': content_ans,
                'content_option': opt_list
            }
    return truth

def normalize_label(label, valid_list):
    def clean(s):
        return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
    cleaned_label = clean(label)
    for v in valid_list:
        if clean(v) == cleaned_label:
            return v
    return "INVALID"
def fuzzy_match(a, b, max_len=None):
    if a == '-' and b == '-':
        return True
    if a == '...' and b == '-':
        return True
    if a == 'None...' and b == '-':
        return True
    if a == 'None' and b == '-':
        return True
    if a in ['ALL', 'None'] or b in ['ALL', 'None']:
        return False

    def clean(s):
        s = str(s)
        s = html.unescape(s)
        s = re.sub(r'\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}', '', s)
        s = re.sub(r'https?://\S+|www\.\S+', '', s)
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    clean_a = clean(a)
    clean_b = clean(b)
    if max_len is not None:
        clean_a = clean_a[:max_len]
        clean_b = clean_b[:max_len]
    if clean_a == clean_b:
        return True
    min_len = 10
    if len(clean_b) >= min_len and clean_b in clean_a:
        return True
    if len(clean_a) >= min_len and clean_a in clean_b:
        return True
    if len(clean_a) >= min_len and len(clean_b) >= min_len:
        similarity = SequenceMatcher(None, clean_a, clean_b).ratio()
        if similarity > 0.85:
            return True
    return False


def evaluate_file(json_path, ground_truth, verbose=True, max_samples=None):
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        if verbose:
            print(f"Limiting to first {max_samples} samples")

    valid_objects = ['tweet_A', 'tweet_B', 'tweet_C', '-']
    valid_types = ['post', 'retweet', 'reply', 'quote']

    y_true_obj, y_pred_obj = [], []
    y_true_type, y_pred_type = [], []
    y_true_cont, y_pred_cont = [], []
    y_true_all, y_pred_all = [], []

    stats = {
        'total_samples': 0,
        'skipped_empty': 0,
        'failed_extract': 0,
        'invalid_object': 0,
        'invalid_type': 0,
        'invalid_content': 0,
        'valid_samples': 0
    }

    if verbose:
        print(f"\nDetailed results ({os.path.basename(json_path)}):")
        print("-" * 140)
        print(f"{'ID':<25} {'Pred Object':<12} {'True Object':<12} {'Obj OK':<6} "
              f"{'Pred Type':<10} {'True Type':<10} {'Type OK':<6} {'Cont OK':<6} "
              f"{'Pred Content (30)':<35} {'True Content'}")
        print("-" * 140)

    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        id_val = list(item.keys())[0]
        if id_val not in ground_truth:
            continue
        stats['total_samples'] += 1
        true_data = ground_truth[id_val]
        true_obj = true_data['action_object_answer']
        true_type = true_data['action_type_answer']
        true_cont = str(true_data['action_content_answer']).strip()
        content_options = true_data['content_option']

        # normalize_label
        true_obj_norm = normalize_label(true_obj, valid_objects)
        true_type_norm = normalize_label(true_type, valid_types)

        output_text = item[id_val].get('output', '')
        if not output_text:
            stats['skipped_empty'] += 1
            if verbose:
                print(f"{id_val:<25} {'EMPTY':<12} {'':<12} {'':<6} {'':<10} {'':<10} {'':<6} {'':<6}")
            continue

        pred_tup = extract_action_from_output(output_text)
        if pred_tup[0] is None:
            stats['failed_extract'] += 1
            if verbose:
                print(f"{id_val:<25} {'EXTRACT_FAIL':<12} {'':<12} {'':<6} {'':<10} {'':<10} {'':<6} {'':<6}")
            continue

        pred_obj_norm = normalize_label(pred_tup[0], valid_objects)
        pred_type_norm = normalize_label(pred_tup[1], valid_types)

        pred_content_label = "INVALID"
        for idx, opt in enumerate(content_options):
            if fuzzy_match(pred_tup[2], opt, 40):
                pred_content_label = str(idx) if idx < 3 else '-'
                break

        is_obj_invalid = (pred_obj_norm == "INVALID")
        is_type_invalid = (pred_type_norm == "INVALID")
        is_content_invalid = (pred_content_label == "INVALID")

        if is_obj_invalid:
            stats['invalid_object'] += 1
        if is_type_invalid:
            stats['invalid_type'] += 1
        if is_content_invalid:
            stats['invalid_content'] += 1

        # skip invalid
        if is_obj_invalid or is_type_invalid or is_content_invalid:
            if verbose:
                print(f"{id_val:<25} {pred_tup[0]:<12} {true_obj:<12} {'INV':<6} "
                      f"{pred_tup[1]:<10} {true_type:<10} {'INV':<6} {'INV':<6} "
                      f"{pred_tup[2][:30]:<35} {str(content_options[int(true_cont)] if true_cont.isdigit() else '-')[:30]}")
            continue

        # valid_samples
        stats['valid_samples'] += 1

        obj_match = (pred_obj_norm == true_obj_norm)
        type_match = (pred_type_norm == true_type_norm)
        content_match = (pred_content_label == true_cont)

        y_true_obj.append(true_obj_norm)
        y_pred_obj.append(pred_obj_norm)
        y_true_type.append(true_type_norm)
        y_pred_type.append(pred_type_norm)
        y_true_cont.append(true_cont)
        y_pred_cont.append(pred_content_label)
        y_true_all.append("True")
        y_pred_all.append(str(obj_match and type_match and content_match))

        if verbose:
            pred_content_short = pred_tup[2][:30] + "..." if len(pred_tup[2]) > 30 else pred_tup[2]
            true_cont_display = str(content_options[int(true_cont)] if true_cont.isdigit() else "-")[:30]
            print(f"{id_val:<25} {pred_tup[0]:<12} {true_obj:<12} {str(obj_match):<6} "
                  f"{pred_tup[1]:<10} {true_type:<10} {str(type_match):<6} {str(content_match):<6} "
                  f"{pred_content_short:<35} {true_cont_display}")

    if verbose:
        print("-" * 140)

    valid_n = stats['valid_samples']
    if valid_n == 0:
        metrics = {k: 0 for k in ['valid_samples', 'object_Accuracy', 'object_Precision', 'object_Recall', 'object_F1',
                                  'type_Accuracy', 'type_Precision', 'type_Recall', 'type_F1',
                                  'content_Accuracy', 'content_Precision', 'content_Recall', 'content_F1',
                                  'overall_Accuracy']}
    else:
        metrics = {
            'valid_samples': valid_n,
            'object_Accuracy': accuracy_score(y_true_obj, y_pred_obj),
            'object_Precision': precision_score(y_true_obj, y_pred_obj, average='macro', zero_division=0),
            'object_Recall': recall_score(y_true_obj, y_pred_obj, average='macro', zero_division=0),
            'object_F1': f1_score(y_true_obj, y_pred_obj, average='macro', zero_division=0),
            'type_Accuracy': accuracy_score(y_true_type, y_pred_type),
            'type_Precision': precision_score(y_true_type, y_pred_type, average='macro', zero_division=0),
            'type_Recall': recall_score(y_true_type, y_pred_type, average='macro', zero_division=0),
            'type_F1': f1_score(y_true_type, y_pred_type, average='macro', zero_division=0),
            'content_Accuracy': accuracy_score(y_true_cont, y_pred_cont),
            'content_Precision': precision_score(y_true_cont, y_pred_cont, average='macro', zero_division=0),
            'content_Recall': recall_score(y_true_cont, y_pred_cont, average='macro', zero_division=0),
            'content_F1': f1_score(y_true_cont, y_pred_cont, average='macro', zero_division=0),
            'overall_Accuracy': accuracy_score(y_true_all, y_pred_all),
        }
    # metrics
    for k in ['total_samples', 'skipped_empty', 'failed_extract', 'invalid_object', 'invalid_type', 'invalid_content']:
        metrics[k] = stats[k]
    return metrics


def main():
    events = ['BlackLivesMatter', 'Covid', 'MeToo', 'Wildfire']

    model = 'deepseek-v4-flash'
    methods = ['MindStep', 'COT', 'Direct_output']

    data_root = r'..\data\SocialAct'
    output_root = r'..\outputs\SocialAct'
    verbose = False
    max_samples = 1000   # limit num

    all_results = []


    for method in methods:
        for event in events:
            json_filename = f"{event}_{method}.json"
            json_path = os.path.join(output_root, model, json_filename)
            if not os.path.exists(json_path):
                print(f"Warning: File not found: {json_path}")
                continue

            csv_path = os.path.join(data_root, f"{event}_1000.csv")
            if not os.path.exists(csv_path):
                print(f"Error: CSV not found: {csv_path}")
                continue

            ground_truth = load_ground_truth(csv_path)
            print(f"\nLoaded {len(ground_truth)} ground truth samples for {event}")

            print(f"\n===== Evaluating: {json_filename} =====")
            metrics = evaluate_file(json_path, ground_truth, verbose=verbose, max_samples=max_samples)


            print(f"Total samples processed: {metrics['total_samples']}")
            stats_to_print = [
                ('skipped_empty', 'Skipped (empty output)'),
                ('failed_extract', 'Failed extraction'),
                ('invalid_object', 'Invalid object'),
                ('invalid_type', 'Invalid type'),
                ('invalid_content', 'Invalid content'),
            ]
            for key, label in stats_to_print:
                val = metrics[key]
                if val > 0:
                    print(f"{label}: {val}")

            valid_n = metrics['valid_samples']
            print(f"Valid samples (all fields valid): {valid_n}")
            if valid_n > 0:
                print(f"Object - Acc: {metrics['object_Accuracy']:.4f}, P: {metrics['object_Precision']:.4f}, R: {metrics['object_Recall']:.4f}, F1: {metrics['object_F1']:.4f}")
                print(f"Type   - Acc: {metrics['type_Accuracy']:.4f}, P: {metrics['type_Precision']:.4f}, R: {metrics['type_Recall']:.4f}, F1: {metrics['type_F1']:.4f}")
                print(f"Content- Acc: {metrics['content_Accuracy']:.4f}, P: {metrics['content_Precision']:.4f}, R: {metrics['content_Recall']:.4f}, F1: {metrics['content_F1']:.4f}")
                print(f"Overall Accuracy: {metrics['overall_Accuracy']:.4f}")
            else:
                print("No valid samples to evaluate.")

            record = {'Event': event, 'Method': method, 'Model': model}
            record.update(metrics)
            all_results.append(record)


    if all_results:
        df = pd.DataFrame(all_results)
        all_stat_cols = [
            'total_samples', 'skipped_empty', 'failed_extract',
            'invalid_object', 'invalid_type', 'invalid_content', 'valid_samples',
            'object_Accuracy', 'object_Precision', 'object_Recall', 'object_F1',
            'type_Accuracy', 'type_Precision', 'type_Recall', 'type_F1',
            'content_Accuracy', 'content_Precision', 'content_Recall', 'content_F1',
            'overall_Accuracy'
        ]
        zero_cols = []
        for col in all_stat_cols:
            if col in df.columns and (df[col] == 0).all():
                zero_cols.append(col)
        if zero_cols:
            df = df.drop(columns=zero_cols)
            # print(f"\nRemoved columns that were all zeros: {zero_cols}")
        #save to csv
        # output_csv = os.path.join(os.getcwd(), f'{model}_evaluation_results.csv')
        # df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        # print(f"\nAll evaluation results saved to: {output_csv}")


if __name__ == "__main__":
    main()

