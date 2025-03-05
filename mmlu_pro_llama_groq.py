import os
import groq
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse

# Set a fixed random seed for reproducibility
random.seed(12345)

def get_client(model_name):
    """
    Initialize Groq client with the specified Llama model
    """
    client = groq.Groq(
        api_key=os.environ.get("GROQ_API_KEY")  # Recommend using environment variable
    )
    return client

def call_api(client, instruction, inputs):
    """
    Send API request to Groq
    """
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Groq's Llama 8B model
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": inputs}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        result = response.choices[0].message.content
        print("cost time", time.time() - start)
        return result
    except Exception as e:
        print(f"API call error: {e}")
        return None

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def single_request(client, single_question, cot_examples_dict):
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None
    pred = extract_answer(response)
    return pred, response

def evaluate(model_name, subjects):
    # Check if Groq API key is set
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Please set the GROQ_API_KEY environment variable")

    client = get_client(model_name)
    test_df, dev_df = load_mmlu_pro()
    
    # If no subjects specified, use all
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Overall results tracking
    overall_results = {
        'total_questions': 0,
        'correct_questions': 0,
        'subject_accuracies': {}
    }
    
    for subject in subjects:
        # Get test data for the subject
        test_data = test_df[subject]
        
        # Randomly sample 100 questions (or all if fewer than 100)
        sampled_test_data = random.sample(test_data, min(100, len(test_data)))
        
        # Tracking for this subject
        subject_results = {
            'total_questions': 0,
            'correct_questions': 0
        }
        
        # Results list for this subject
        subject_output_results = []
        
        for each in tqdm(sampled_test_data):
            label = each["answer"]
            pred, response = single_request(client, each, dev_df)
            
            # Prepare result entry
            result_entry = each.copy()
            result_entry['pred'] = pred
            result_entry['model_outputs'] = response
            subject_output_results.append(result_entry)
            
            # Update tracking
            subject_results['total_questions'] += 1
            overall_results['total_questions'] += 1
            
            # Check prediction accuracy
            if pred == label:
                subject_results['correct_questions'] += 1
                overall_results['correct_questions'] += 1
        
        # Calculate subject accuracy
        subject_accuracy = subject_results['correct_questions'] / subject_results['total_questions']
        overall_results['subject_accuracies'][subject] = {
            'accuracy': subject_accuracy,
            'total_questions': subject_results['total_questions'],
            'correct_questions': subject_results['correct_questions']
        }
        
        # Save subject results
        output_res_path = os.path.join(args.output_dir, f"{model_name}_{subject}_sampled_result.json")
        with open(output_res_path, "w") as fo:
            json.dump(subject_output_results, fo, indent=2)
    
    # Calculate overall accuracy
    overall_accuracy = overall_results['correct_questions'] / overall_results['total_questions']
    overall_results['overall_accuracy'] = overall_accuracy
    
    # Save overall results summary
    output_summary_path = os.path.join(args.output_dir, f"{model_name}_sampled_summary.json")
    with open(output_summary_path, "w") as fo:
        json.dump(overall_results, fo, indent=2)
    
    # Print results
    print("\nSubject-wise Accuracies:")
    for subject, details in overall_results['subject_accuracies'].items():
        print(f"{subject}: {details['accuracy']:.2%} ({details['correct_questions']}/{details['total_questions']})")
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({overall_results['correct_questions']}/{overall_results['total_questions']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="./output")
    parser.add_argument("--model_name", "-m", type=str, default="llama3-8b-8192",
                        choices=["llama3-8b-8192"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")

    args, unknown = parser.parse_known_args()

    assigned_subjects = []
    if args.assigned_subjects != "all":
        assigned_subjects = args.assigned_subjects.split(",")

    # Run evaluation
    evaluate(args.model_name, assigned_subjects)