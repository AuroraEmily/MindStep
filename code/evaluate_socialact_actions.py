# -*- coding: utf-8 -*-
import os
import json
import csv
import re
import ast
import warnings
import html
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from difflib import SequenceMatcher

warnings.filterwarnings('ignore', category=RuntimeWarning)




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
        if SequenceMatcher(None, clean_a, clean_b).ratio() > 0.85:
            return True
    return False


def normalize_label(label, valid_list):
    def clean(s):
        return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
    cleaned_label = clean(label)
    for v in valid_list:
        if clean(v) == cleaned_label:
            return v
    return "INVALID"




def extract_action_from_output(output_text):
    if not output_text or not isinstance(output_text, str):
        return None, None, None
    if "--- Final Answer ---\nERROR: failed to parse model output" in output_text:
        return None, None, None

    def clean_value(v):
        if v is None:
            return None
        v = str(v)
        v = re.sub(r'<[^>]+>', '', v)
        v = v.replace('**', '').strip().strip('"').strip("'").strip()
        v = re.sub(r'^\s*\\*\s*', '', v).strip()
        v = re.sub(r'[,\\.;\s]+$', '', v).strip()
        if re.match(r'^\s*["\']?\s*-\s*["\']?', v):
            return '-'
        if v.lower() in ['none', 'null', 'n/a']:
            return '-'
        return v

    def clean_obj(v):
        v = clean_value(v)
        if v is None:
            return None
        if re.search(r'^\s*["\']?\s*-\s*["\']?\s*(?:\([^)]*\))?\s*$', v):
            return '-'
        m = re.search(r'\btweet[_\s-]*([ABC])\b', v, re.I)
        return f"tweet_{m.group(1).upper()}" if m else v

    def clean_type(v):
        v = clean_value(v)
        if v is None:
            return None
        m = re.search(r'\b(post|retweet|reply|quote)\b', v, re.I)
        return m.group(1).lower() if m else v

    text = output_text
    answer_blocks = re.findall(r'<Answer>(.*?)</Answer>', text, flags=re.I | re.S)
    search_text = answer_blocks[-1].strip() if answer_blocks else text


    try:
        data = json.loads(search_text.strip())
        typ = data.get("Action Type") or data.get("action_type") or data.get("type")
        obj = data.get("Action Object") or data.get("action_object") or data.get("object")
        cont = data.get("Action Content") or data.get("action_content") or data.get("content")
        if typ is not None and obj is not None and cont is not None:
            return clean_obj(obj), clean_type(typ), clean_value(cont)
    except Exception:
        pass


    field_pat = re.compile(
        r'(?:\(\d+\)\s*)?["\']?\s*Action\s+(Type|Object|Content)\s*["\']?\s*[:：]\s*',
        re.I
    )
    matches = list(field_pat.finditer(search_text))
    if matches:
        vals = {}
        for idx, m in enumerate(matches):
            key = m.group(1).lower()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(search_text)
            val = search_text[start:end]
            val = re.sub(r'^[,\\.;\s]+', '', val).strip()
            val = re.sub(r'</Answer>.*$', '', val, flags=re.I | re.S).strip()
            vals[key] = clean_value(val)
        if vals.get('type') is not None and vals.get('object') is not None and vals.get('content') is not None:
            return clean_obj(vals['object']), clean_type(vals['type']), clean_value(vals['content'])


    if search_text:
        m = re.match(
            r'^\s*(post|retweet|reply|quote)\s*[,;|]\s*(tweet[_\s-]?[ABC]|-)\s*[,;|]\s*(.*?)\s*$',
            search_text, flags=re.I | re.S
        )
        if m:
            return clean_obj(m.group(2)), clean_type(m.group(1)), clean_value(m.group(3))


    lines = [clean_value(x) for x in search_text.splitlines() if clean_value(x)]
    if len(lines) >= 3 and re.fullmatch(r'(post|retweet|reply|quote)', lines[0], re.I):
        return clean_obj(lines[1]), clean_type(lines[0]), clean_value(' '.join(lines[2:]))


    m = re.search(r'\b(post|retweet|reply|quote)\b\s+(?:to|of)?\s*(tweet[_\s-]?[ABC]|-)', text, re.I)
    if m:
        return clean_obj(m.group(2)), clean_type(m.group(1)), '-'


    actions = re.findall(r'\b(post|retweet|reply|quote)\b', text, re.I)
    tweets = re.findall(r'\btweet[_\s-]*([ABC])\b', text, re.I)
    if actions:
        typ = actions[-1].lower()
        obj = f"tweet_{tweets[-1].upper()}" if tweets else '-'
        return obj, typ, '-'
    return None, None, None


# ==================== 加载 ground truth ====================

def load_ground_truth(csv_path):
    truth = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required_cols = ['id', 'action_object_answer', 'action_type_answer',
                         'action_content_answer', 'content_option']
        for col in required_cols:
            if col not in reader.fieldnames:
                raise KeyError(f"CSV missing '{col}' column, columns: {reader.fieldnames}")
        for row in reader:
            id_val = row['id'].strip()
            obj_ans = row['action_object_answer'].strip()
            type_ans = row['action_type_answer'].strip()
            content_raw = row['action_content_answer'].strip()
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
                'action_content_answer': content_raw,
                'content_option': opt_list
            }
    return truth


# ==================== 评估函数 ====================

def evaluate_file(json_path, ground_truth, verbose=True, max_samples=None):
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        if verbose:
            print(f"Limiting to first {max_samples} samples")

    stats = {'total': 0, 'skipped_empty': 0, 'failed_extract': 0}
    y_true_obj, y_pred_obj = [], []
    y_true_type, y_pred_type = [], []
    y_true_cont, y_pred_cont = [], []
    y_true_all, y_pred_all = [], []

    valid_objects = ['tweet_A', 'tweet_B', 'tweet_C', '-']
    valid_types = ['post', 'retweet', 'reply', 'quote']

    if verbose:
        print(f"\nDetailed results ({os.path.basename(json_path)}):")
        print("-" * 140)
        print(f"{'ID':<25} {'Pred Obj':<12} {'True Obj':<12} {'Obj OK':<6} "
              f"{'Pred Type':<10} {'True Type':<10} {'Type OK':<6} {'Cont OK':<6} "
              f"{'Pred Content (30)':<35} {'True Content'}")
        print("-" * 140)

    for item in data:
        for id_val, content in item.items():
            if id_val not in ground_truth:
                continue
            true_data = ground_truth[id_val]
            true_obj = true_data['action_object_answer']
            true_type = true_data['action_type_answer']
            true_cont = str(true_data['action_content_answer']).strip()
            content_options = true_data['content_option']

            output_text = content.get('output', '')
            if not output_text:
                stats['skipped_empty'] += 1
                if verbose:
                    print(f"{id_val:<25} {'EMPTY':<12}")
                continue

            stats['total'] += 1


            pred_tup = extract_action_from_output(output_text)
            if pred_tup[0] is None:
                stats['failed_extract'] += 1
                pred_tup = ('INVALID', 'INVALID', 'INVALID')


            pred_obj_norm = normalize_label(pred_tup[0], valid_objects)
            true_obj_norm = normalize_label(true_obj, valid_objects)
            tweet_match = (pred_obj_norm == true_obj_norm) and (pred_obj_norm != "INVALID")
            y_pred_obj.append(pred_obj_norm)
            y_true_obj.append(true_obj_norm)


            pred_type_norm = normalize_label(pred_tup[1], valid_types)
            true_type_norm = normalize_label(true_type, valid_types)
            action_match = (pred_type_norm == true_type_norm) and (pred_type_norm != "INVALID")
            y_pred_type.append(pred_type_norm)
            y_true_type.append(true_type_norm)


            pred_content_label = "INVALID"

            option_match = re.search(r'\boption\s*([1-4])\b', str(pred_tup[2]), re.I)
            if option_match:
                option_num = int(option_match.group(1))
                pred_content_label = ['0', '1', '2', '-'][option_num - 1] if 1 <= option_num <= 4 else "INVALID"
            else:
                for idx, option in enumerate(content_options):
                    if fuzzy_match(pred_tup[2], option, 40):
                        pred_content_label = ['0', '1', '2', '-'][idx] if idx < 3 else '-'
                        break

            content_match = (pred_content_label == true_cont) and (pred_content_label != "INVALID")
            y_pred_cont.append(pred_content_label)
            y_true_cont.append(true_cont)


            all_match = tweet_match and action_match and content_match
            y_pred_all.append(str(all_match))
            y_true_all.append("True")

            if verbose:
                pred_content_short = pred_tup[2][:30] + "..." if len(pred_tup[2]) > 30 else pred_tup[2]
                true_cont_display = str(content_options[int(true_cont)] if true_cont.isdigit() else "-")[:30]
                print(f"{id_val:<25} {pred_tup[0]:<12} {true_obj:<12} {str(tweet_match):<6} "
                      f"{pred_tup[1]:<10} {true_type:<10} {str(action_match):<6} {str(content_match):<6} "
                      f"{pred_content_short:<35} {true_cont_display}")

    if verbose:
        print("-" * 140)

    total = stats['total']
    invalid_cnt = sum(p == 'INVALID' for p in y_pred_cont)

    if total == 0:
        metrics = {k: 0.0 for k in [
            'valid_samples', 'object_Accuracy', 'object_Precision', 'object_Recall', 'object_F1',
            'type_Accuracy', 'type_Precision', 'type_Recall', 'type_F1',
            'content_Accuracy', 'content_Precision', 'content_Recall', 'content_F1',
            'overall_Accuracy'
        ]}
        metrics['content_INVALID_count'] = 0
    else:
        metrics = {
            'valid_samples': total,
            'object_Accuracy': accuracy_score(y_true_obj, y_pred_obj),
            'object_Precision': precision_score(y_true_obj, y_pred_obj, labels=valid_objects, average='macro', zero_division=0),
            'object_Recall': recall_score(y_true_obj, y_pred_obj, labels=valid_objects, average='macro', zero_division=0),
            'object_F1': f1_score(y_true_obj, y_pred_obj, labels=valid_objects, average='macro', zero_division=0),
            'type_Accuracy': accuracy_score(y_true_type, y_pred_type),
            'type_Precision': precision_score(y_true_type, y_pred_type, labels=valid_types, average='macro', zero_division=0),
            'type_Recall': recall_score(y_true_type, y_pred_type, labels=valid_types, average='macro', zero_division=0),
            'type_F1': f1_score(y_true_type, y_pred_type, labels=valid_types, average='macro', zero_division=0),
            'type_F1_Weighted': f1_score(y_true_type, y_pred_type, labels=valid_types, average='weighted', zero_division=0),
            'content_Accuracy': accuracy_score(y_true_cont, y_pred_cont),
            'content_Precision': precision_score(y_true_cont, y_pred_cont, labels=['0', '1', '2', '-'], average='macro', zero_division=0),
            'content_Recall': recall_score(y_true_cont, y_pred_cont, labels=['0', '1', '2', '-'], average='macro', zero_division=0),
            'content_F1': f1_score(y_true_cont, y_pred_cont, labels=['0', '1', '2', '-'], average='macro', zero_division=0),
            'content_INVALID_count': invalid_cnt,
            'overall_Accuracy': accuracy_score(y_true_all, y_pred_all),
        }

    metrics['skipped_empty'] = stats['skipped_empty']
    metrics['failed_extract'] = stats['failed_extract']
    return metrics


# ==================== 主函数 ====================

def main():
    events = ['BlackLivesMatter', 'Covid', 'MeToo', 'Wildfire']
    model = 'deepseek-v4-flash'
    methods = ['MindStep', 'COT', 'Direct_output']

    data_root = r'..\data\SocialAct'
    output_root = r'..\outputs\SocialAct'
    verbose = False
    max_samples = 1000

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

            print(f"Total samples processed: {metrics['valid_samples']}")
            print(f"Skipped (empty output): {metrics['skipped_empty']}")
            print(f"Failed extraction: {metrics['failed_extract']}")

            if metrics['valid_samples'] > 0:
                print(f"Object  - Acc: {metrics['object_Accuracy']:.4f}, P: {metrics['object_Precision']:.4f}, "
                      f"R: {metrics['object_Recall']:.4f}, F1: {metrics['object_F1']:.4f}")
                print(f"Type    - Acc: {metrics['type_Accuracy']:.4f}, P: {metrics['type_Precision']:.4f}, "
                      f"R: {metrics['type_Recall']:.4f}, F1: {metrics['type_F1']:.4f}")
                print(f"Content - Acc: {metrics['content_Accuracy']:.4f}, P: {metrics['content_Precision']:.4f}, "
                      f"R: {metrics['content_Recall']:.4f}, F1: {metrics['content_F1']:.4f} "
                      f"(INVALID: {metrics['content_INVALID_count']})")
                print(f"Overall Accuracy: {metrics['overall_Accuracy']:.4f}")
            else:
                print("No valid samples to evaluate.")

            record = {'Event': event, 'Method': method, 'Model': model}
            record.update(metrics)
            all_results.append(record)

    if all_results:
        df = pd.DataFrame(all_results)

        all_stat_cols = [
            'valid_samples', 'skipped_empty', 'failed_extract',
            'object_Accuracy', 'object_Precision', 'object_Recall', 'object_F1',
            'type_Accuracy', 'type_Precision', 'type_Recall', 'type_F1', 'type_F1_Weighted',
            'content_Accuracy', 'content_Precision', 'content_Recall', 'content_F1',
            'content_INVALID_count', 'overall_Accuracy'
        ]
        zero_cols = []
        for col in all_stat_cols:
            if col in df.columns and (df[col] == 0).all():
                zero_cols.append(col)
        if zero_cols:
            df = df.drop(columns=zero_cols)

        output_csv = os.path.join(os.getcwd(), f'{model}_evaluation_results.csv')
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nAll evaluation results saved to: {output_csv}")


if __name__ == "__main__":
    main()