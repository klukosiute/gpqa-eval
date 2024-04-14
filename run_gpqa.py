import anthropic
from openai import OpenAI
import pandas as pd
import random
import json
import pickle
import re
import argparse
import os


def parse_sampled_answer(answer, answer_choice_tokens):
    """Copy pasted from GPQA repo + more patterns"""
    patterns = [
        r"answer is \((.)\)",
        r"Answer: \((.)\)",
        r"answer: \((.)\)",
        r"answer \((.)\)",
        r"answer is (\w)\.",
        r"Answer: (\w)\.",
        r"answer: (\w)\.",
        r"answer (\w)\.",
        r"answer is option \((.)\)",
        r"answer is Option \((.)\)",
        r"answer is option (\w)",
        r"answer is Option (\w)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in answer_choice_tokens:
            return match.group(1)
    return None


def format_gpqa(filepath="/Users/kamile/code/gpqa_diamond.csv"):
    df = pd.read_csv(filepath)
    question_format = """{}
(A) {}
(B) {}
(C) {}
(D) {}

After your reasoning, provide your final, single letter choice, formatted as "The answer is (X)." """

    questions = df["Question"].to_list()
    correct_answers = df["Correct Answer"].to_list()

    all_answers = [
        list(x)
        for x in list(
            zip(
                *[
                    df["Correct Answer"].to_list(),
                    df["Incorrect Answer 1"].to_list(),
                    df["Incorrect Answer 2"].to_list(),
                    df["Incorrect Answer 3"].to_list(),
                ]
            )
        )
    ]
    [random.shuffle(x) for x in all_answers]

    correct_letters = []
    for correct, all_answer in zip(correct_answers, all_answers):
        correct_letters.append("ABCD"[all_answer.index(correct)])

    formatted_prompts = [
        question_format.format(*[question, *answers])
        for question, answers in zip(questions, all_answers)
    ]

    return correct_letters, formatted_prompts


def load_formatted_gpqa(corrects_path, prompts_path):
    # ew sorry
    with open(corrects_path, "rb") as file:
        # Load the pickled list from the file
        corrects = pickle.load(file)

    with open(prompts_path, "rb") as file:
        # Load the pickled list from the file
        prompts = pickle.load(file)

    return corrects, prompts


def run_eval_claude(client, model, run_id, file_to_dump):
    print("Running GPQA on {}, {}th iteration.".format(model, run_id))

    if os.path.exists(file_to_dump):
        with open(file_to_dump, "r") as file:
            result = json.load(file)
        prompts = result["prompts"]
        corrects = result["corrects"]
        start_index = len(result["grading"])
    else:
        corrects, prompts = format_gpqa()
        result = {
            "model": model,
            "run_id": run_id,
            "corrects": corrects,
            "prompts": prompts,
            "grading": [],
            "sampled_texts": [],
        }
        start_index = 0
    print(start_index)
    for i in range(start_index, len(prompts)):
        prompt = prompts[i]
        correct_letter = corrects[i]

        if i % 20 == 0:
            print(i)

        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Let's think step by step:"}],
                },
            ],
        )
        result["sampled_texts"].append(message.content[0].text)
        result["grading"].append(
            parse_sampled_answer(message.content[0].text, list("ABCD"))
            == correct_letter
        )

        if (i + 1) % 10 == 0:
            with open(file_to_dump, "w") as file:
                json.dump(result, file)

    with open(file_to_dump, "w") as file:
        json.dump(result, file)


def run_eval_openai(client, model, run_id, file_to_dump):
    print("Running GPQA on {}, {}th iteration.".format(model, run_id))

    if os.path.exists(file_to_dump):
        with open(file_to_dump, "r") as file:
            result = json.load(file)
        prompts = result["prompts"]
        corrects = result["corrects"]
        start_index = len(result["grading"])
    else:
        corrects, prompts = format_gpqa()
        result = {
            "model": model,
            "run_id": run_id,
            "corrects": corrects,
            "prompts": prompts,
            "grading": [],
            "sampled_texts": [],
        }
        start_index = 0
    print(start_index)
    for i in range(start_index, len(prompts)):
        prompt = prompts[i]
        correct_letter = corrects[i]

        if i % 20 == 0:
            print(i)

        message = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1000,
        )

        result["sampled_texts"].append(message.choices[0].message.content)
        result["grading"].append(
            parse_sampled_answer(message.choices[0].message.content, list("ABCD"))
            == correct_letter
        )

        if (i + 1) % 10 == 0:
            with open(file_to_dump, "w") as file:
                json.dump(result, file)

    with open(file_to_dump, "w") as file:
        json.dump(result, file)


def main():
    parser = argparse.ArgumentParser(
        description="Run GPQA evaluation on a specified model."
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="Provider of the model (e.g., 'anthropic', 'openai').",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to evaluate."
    )
    parser.add_argument(
        "--run_ids",
        type=int,
        nargs="+",
        required=True,
        help="List of run IDs for the evaluation.",
    )
    args = parser.parse_args()
    run_ids = args.run_ids
    provider = args.provider
    model = args.model

    if provider == "anthropic":
        with open("/Users/kamile/code/oai/.anthropic.secret", "r") as f:
            api_key = f.read().strip()

        # api_key = contents.split("\n")[0].split("=")[1].strip()
        client = anthropic.Anthropic(
            api_key=api_key,
        )

        for run_id in run_ids:
            output_path = "-".join([provider, str(run_id), model, "gpqa"]) + ".json"
            run_eval_claude(client, model, run_id, output_path)

    elif provider == "openai":
        with open("/Users/kamile/code/oai/.openai.secret", "r") as f:
            contents = f.read().strip()
        api_key = contents.split("\n")[0].split("=")[1].strip()
        client = OpenAI(api_key=api_key)

        for run_id in run_ids:
            output_path = "-".join([provider, str(run_id), model, "gpqa"]) + ".json"
            run_eval_openai(client, model, run_id, output_path)

    else:
        print(f"Provider '{provider}' not supported.")


if __name__ == "__main__":

    main()
