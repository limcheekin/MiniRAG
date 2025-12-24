import json
import sys
from pathlib import Path

def transform_query_set_to_eval_dataset(query_set_path, output_path):
    with open(query_set_path, 'r', encoding='utf-8') as f:
        query_set = json.load(f)

    test_cases = []
    for key, value in query_set.items():
        # Defensive: skip non-dict entries
        if not isinstance(value, dict):
            continue
        question = value.get('question')
        answer = value.get('answer')
        project = value.get('type')
        if question is not None and answer is not None and project is not None:
            test_cases.append({
                'question': question,
                'ground_truth': answer,
                'project': project
            })

    output = {'test_cases': test_cases}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <query_set.json> <output.json>")
        sys.exit(1)
    transform_query_set_to_eval_dataset(sys.argv[1], sys.argv[2])
