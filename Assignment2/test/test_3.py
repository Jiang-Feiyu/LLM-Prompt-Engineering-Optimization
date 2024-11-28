api_key="50067a25-be0a-416b-83ea-bf53f65f4cb7"

from openai import OpenAI
import json
import re
import time
import sys
import transformers
import torch

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=api_key)

zero_shot_prompt = [{'role': 'system', 'content': 'Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.'}, {'role': 'user', 'content': ''}]

few_shot_prompt = [{'role': 'system', 'content': 'Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.'}, {'role': 'user', 'content': 'Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = <<21-15=6>>6 trees that were planted.\n#### 6"}, {'role': 'user', 'content': 'Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = <<3+2=5>>5 cars are in the parking lot.\n#### 5"}, {'role': 'user', 'content': 'Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nOriginally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left in total.\n#### 39"}, {'role': 'user', 'content': 'Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nJason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8"}, {'role': 'user', 'content': 'Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nShawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = <<2*2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9"}, {'role': 'user', 'content': 'Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = <<4*5=20>>20 computers were added. Now 9 + 20 = <<9+20=29>>29 computers are now in the server room.\n#### 29"}, {'role': 'user', 'content': 'Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nMichael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = <<58-23=35>>35 at the end of Tuesday, and 35 - 2 = <<35-2=33>>33 at the end of wednesday.\n#### 33"}, {'role': 'user', 'content': 'Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nOlivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. Now she has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8"}, {'role': 'user', 'content': 'Question: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?'}]

few_shot_prompt_correctdemo_wrongans = []

def load_json_objects(filename):
    json_list = []
    
    with open(filename, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())  # 去掉行末的换行符并加载 JSON
            json_list.append(json_obj)
    
    return json_list

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def delete_extra_zero(n):
    '''Delete the extra 0 after the decimal point'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n

def extract_ans_from_response(answer: str, eos=None):
    '''
    :param answer: model-predicted solution or golden answer string
    :param eos: stop token
    :return:
    '''
    if eos:
        answer = answer.split(eos)[0].strip()
    answer = answer.split('####')[-1].strip()
    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

def send_request(msg):
    try:
        completion = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=msg,
            temperature=0.1,
            top_p=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return  "#### -10000000000000"

def generate_weak_ans(question):
    t = []
    # zero_shot_prompt = [{'role': 'system', 'content': 'Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.'}, {'role': 'user', 'content': ''}]
    few_shot_prompt = [{'role': 'system', 'content': 'Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.'}, {'role': 'user', 'content': 'Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = <<21-15=6>>6 trees that were planted.\n#### 6"}, {'role': 'user', 'content': 'Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = <<3+2=5>>5 cars are in the parking lot.\n#### 5"}, {'role': 'user', 'content': 'Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nOriginally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left in total.\n#### 39"}, {'role': 'user', 'content': 'Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nJason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8"}, {'role': 'user', 'content': 'Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nShawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = <<2*2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9"}, {'role': 'user', 'content': 'Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nThere were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = <<4*5=20>>20 computers were added. Now 9 + 20 = <<9+20=29>>29 computers are now in the server room.\n#### 29"}, {'role': 'user', 'content': 'Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nMichael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = <<58-23=35>>35 at the end of Tuesday, and 35 - 2 = <<35-2=33>>33 at the end of wednesday.\n#### 33"}, {'role': 'user', 'content': 'Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'}, {'role': 'assistant', 'content': "Answer:\nLet's think step by step.\nOlivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. Now she has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8"}, {'role': 'user', 'content': 'Question: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?'}]
    t = few_shot_prompt # zero-shot
    t[-1]['content']=question # zero-shot
    weak_ans = send_request(t)
    t.append({'role':'assistant', 'content':weak_ans})
    return t, weak_ans

def is_correct_ans(answer, LLM_ans):
    test_answer = extract_ans_from_response(answer)
    if isinstance(test_answer, str):
        test_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', test_answer)[0]
    test_answer = delete_extra_zero(test_answer)

    llm_answer = extract_ans_from_response(LLM_ans)
        
    if isinstance(llm_answer, str):
        try:
            llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)[0]
        except IndexError:
            llm_answer = -10000000  # 或者你可以设置一个默认值
        except Exception as e:
            print(f"发生错误: {e}")
    llm_answer = delete_extra_zero(llm_answer)

    if test_answer == llm_answer:
        return True
    else:
        return False
        

filename = 'data/GSM8K/test.jsonl'
# log_file_path = "log.jsonl" # zero-shot
log_file_path = "./output/few_shot_prompt_wrong_demo.jsonl"
data_list = load_json_objects(filename)
prompt_path = "./prompt/few_shot_prompt_wrong_demo.json"

# 读取prompt文件
with open(prompt_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

count = 0
total_time = 0

for idx, input in enumerate(data_list):
    # zero shot:
    # temp = zero_shot_prompt # zero-shot
    # temp[1]['content']=input["question"] # zero-shot

    # few shot:
    # temp = few_shot_prompt # few-shot
    # temp[-1]['content']=input["question"] # few-shot

    # correct demo with wrong answer:
    temp = data
    temp[-1]['content']=input["question"] # few-shot correct demo with wrong answer

    start_time = time.time()  # Start timing

    res = send_request(temp)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    total_time += elapsed_time  # Add to total time

    # 获取标准答案
    test_answer = extract_ans_from_response(input["answer"]) # ans from dataset
    if isinstance(test_answer, str):
        test_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', test_answer)[0]
    test_answer = delete_extra_zero(test_answer)


    # compare answer
    llm_answer = extract_ans_from_response(res)
    if isinstance(llm_answer, str):
        try:
            llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)
        except IndexError:
            llm_answer = -10000000  
        except Exception as e:
            print(f"发生错误: {e}")

    llm_answer = delete_extra_zero(llm_answer)
    
    if test_answer == llm_answer:
        is_correct = True
    else:
        is_correct = False

    if is_correct == True:
        count += 1

    # 记录到 log.jsonl
    log_entry = {
        "input": input["question"],
        "prompt": temp, 
        "output":res,
        "ans": input["answer"],
        "correct_answer_num": test_answer,  # 记录正确答案
        "response_num": llm_answer,
        "is_correct": is_correct
    }

    with open(log_file_path, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')

    # 实时输出正确率
    accuracy_rate = count / (idx + 1)  # 当前的正确率
    # print(f"\rCurrent accuracy: {accuracy_rate:.2%} ({count}/{idx + 1})", end='')

    # Progress bar
    progress = (idx + 1) / len(data_list)
    bar_length = 40  # Length of the bar in characters
    block = int(round(bar_length * progress))
    progress_bar = "#" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\rProgress: [{progress_bar}] {idx + 1}/{len(data_list)} ({progress * 100:.2f}%) \rCurrent accuracy: {accuracy_rate:.2%} ({count}/{idx + 1})\r")
    sys.stdout.flush()

    time.sleep(3)

accuracy_rate = float(count/len(data_list))

print(f"""Evaluating method combine:
Accuracy: {accuracy_rate} ({count}/1319)
Total inference time (seconds): {total_time}
""")