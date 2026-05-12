import argparse
import os
from copy import deepcopy
import pandas as pd
import numpy as np

# ==========================================
# Prompt Definitions
# ==========================================

# System Instructions (General context for the model)
SYSTEM_PROMPTS = {
    "system_instruction": 
'''You are a helpful assistant.''',

}
# User Instructions (Prompt Templates requiring formatting)
# Note: Format keys like {problem}, {standard_answer}, {thought_process} are expected.
USER_PROMPTS = {
    "se_usr": 
'''
Your task is to take the original expert reasoning and rephrase it using natural and familiar vocabulary while strictly preserving the original semantics, logical steps, and structure. Do not add, remove, or rearrange any content; focus solely on linguistic paraphrasing to make the wording more accessible.

## **Instructions**
1. You must translate and paraphrase every single sentence, equation, and thought from the original text one by one.
2. Do not mention, imply, or acknowledge the existence of an "expert," an "original text," or the act of "translating/rephrasing." The output must read entirely as your own independent, organic thought process.
3. Present the final result as a standard, clear, and complete step-by-step solution to the problem.

## **Input**
[Problem]:
{problem}\n
[Expert's Reasoning Process]:
{thought_process}

## **Output**
/no_think''',

"se_usr_translate":
'''
You are an expert technical translator, specializing in translating text from any language into highly accurate, fluent, and professional English. 
Translate the provided source text into English. Your primary goal is to ensure absolute semantic accuracy while strictly preserving all non-linguistic elements (such as mathematical formulas, code, and markdown formatting).

## **Input**
{thought_process}

## **Output**
/no_think
'''
}
#Please rewrite the following expert reasoning using your most natural and familiar vocabulary. Focus purely on linguistic paraphrasing to make the wording more accessible to you. You must strictly preserve the exact original semantics, logical steps, and structural format without adding, removing, or rearranging anything.

# ==========================================
# Arguments and Main Logic
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset with specified prompts.")
    
    # Defaults based on original script
    default_input = "/home/zhwang/LLM/Train/data_0306/openr1.parquet"
    default_output = "/home/zhwang/LLM/Train/data_0306/se_prompt/openr1_seprompt_translate.parquet"
    
    parser.add_argument("--input_file", type=str, default=default_input, help="Path to the input parquet file.")
    parser.add_argument("--output_file", type=str, default=default_output, help="Path to save the output parquet file.")
    parser.add_argument("--prompt_key", type=str, default="se_usr", choices=USER_PROMPTS.keys(), 
                        help="Key for the user prompt template to use.")
    parser.add_argument("--system_key", type=str, default="system_instruction", choices=SYSTEM_PROMPTS.keys(), 
                        help="Key for the system prompt template to use.")
    return parser.parse_args()

def process_data(dataset, usr_prompt, system_prompt):
    ret_dict = []
    
    for item in dataset:
        prompt = item.get('prompt')
        target = item.get('target')
        system_p = deepcopy(prompt[0])
        
        try:
            problem = prompt[1]['content']
            # Target is expected to be a list with at least one element containing 'content'
            target_solution = target[0]['content']
        except (IndexError, KeyError, TypeError):
            # Skip invalid items
            continue

        # Extract Think Process and Standard Answer
        # Logic: Split by </think>. First part is thought, second is answer.
        # This assumes the input data is formatted with <think>...</think>Answer
        if "</think>" in target_solution:
            parts = target_solution.split("</think>")
            think_process = parts[0].strip()
            # If standard answer exists (len > 1), use it, otherwise empty string or handle gracefully
            if len(parts) > 1:
                standard_answer = parts[1].strip()
            else:
                standard_answer = ""
                
            if "<think>" in think_process:
                think_process = think_process.replace("<think>", "").strip()
        else:
            # If no </think> tag, use content as think_process for now or skip?
            # Original script logic implies </think> is critical.
            continue

        # Format input prompt
        try:
            # Prepare arguments for formatting
            format_kwargs = {
                "problem": problem,
                "thought_process": think_process,
                "standard_answer": standard_answer,
                "question": problem  # Alias for compatibility
            }
            se_input = usr_prompt.format(**format_kwargs)
        except KeyError as e:
            # This happens if the template expects a key (like {standard_answer}) but we passed it empty/None
            # or if the template expects keys we didn't provide.
            # print(f"Skipping item due to missing key for template: {e}")
            continue

        # Create new prompt structure
        system_p['content'] = system_prompt
        item['se_prompt'] = np.array([system_p, {'role': 'user', 'content': se_input}])
        
        # # Modify original prompt/target fields
        # item['prompt'][0]['content'] = system_prompt
        # item['prompt'][1]['content'] = USR_INSTRUCTION_TEMPLATE.format(question=problem)
        # item['target'][0]['content'] = think_process

        # Delete unwanted keys to clean up
        keys_to_delete = [
            "responses", "test_score", "gemini_thinking_trajectory", 
            "gemini_attempt", "deepseek_thinking_trajectory", 
            "deepseek_attempt", "gemini_grade", "gemini_grade_reason", 
            "deepseek_grade_reason"
        ]
        for k in keys_to_delete:
            if k in item:
                del item[k]
        
        ret_dict.append(item)
    
    return ret_dict

def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    print(f"Loading data from: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    dataset = df.to_dict(orient="records")

    prompt_key = args.prompt_key
    system_key = args.system_key
    
    selected_usr_prompt = USER_PROMPTS[prompt_key]
    selected_system_prompt = SYSTEM_PROMPTS[system_key]

    print(f"Using System Prompt: {selected_system_prompt}")
    print(f"Using User Prompt: {selected_usr_prompt}")
    
    print(f"Processing {len(dataset)} items...")

    processed_data = process_data(dataset, selected_usr_prompt, selected_system_prompt)
    
    if len(processed_data) > 0:
        print(f"Processed {len(processed_data)} items.")
        #print("Sample processed item (first one) 'se_prompt':")
        # print(processed_data[0]['se_prompt'])
    else:
        print("No items were processed correctly (possibly due to missing </think> tags or prompt format issues).")

    print(f"Saving output to: {args.output_file}")
    
    # Ensure directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(processed_data)
    train_df.to_parquet(args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()
