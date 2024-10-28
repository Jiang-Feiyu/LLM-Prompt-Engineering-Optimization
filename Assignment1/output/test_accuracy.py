api_key=""

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
        
def self_refine(question, answer):
    def generate_hint(question, weak_answer):
        query = f'''Question: {question}
Now we have a weak answer: 

<weak answer>
{weak_answer}
<weak answer> 

You should generate some hints to improve the answer.

Your creteria should include:

<criterion>
- The final answer must include “#### [value]” format (e.g., “#### 500”).
- Every number used in the steps must be provided by the question. For example, you should check if any number in the weak answer can not be found in the question. The intermediate results calculated through the variables in the question are not counted.
- The answer must directly correspond to the question asked. For example, if the question asks the number of Apple, the answer should be answer the amout of Apple, not banana.
- Each step should have a logical explanation justifying its inclusion (e.g., formulas).
<criterion>

<requirements>
- The hint should be less than 50 words. 
- The hint should not give the answer directly, so avoid any calculation.
- You Should not return improved answer, or comments to the weak answer.
- The hint should be return in bullet points.
- Hint SHOULD NOT provide steps to solve the question.
<requirements>'''
        hints = send_request([{'role':'user', 'content':query}])
        if len(hints) > 3000:
            return hints[0:3000]
        return hints

    def get_better_ans(question, hints, temp):
        query = f'''Here is a Math Question: 
<Question>
{question}
<Question>

<Task>
Your task is to solve a series of math word problems by providing the final answer. 
- Show your answer step by step.
- Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.
<Task>

<Requirements>
- Please answer the question step by step
- Please refer the hint for reference
- The solution you provided should be short and clear
- Each setp should less than 50 words, and you should solve this problem in less than 10 steps
<Requirements>

<Answer>
#### 
<Answer>

Here are some hints for you to reference
<Hint>
{hints}
<Hint>
'''
        if len(query) > 4000:
            temp.append({'role':'user', 'content':query[:4000]})
        else:
            temp.append({'role':'user', 'content':query})
        better_ans = send_request([{'role':'user', 'content':query}])
        temp.append({'role':'assistant', 'content':better_ans})
        return temp, better_ans
     
    # 设置最大迭代次数为3
    max_iter = 3
    temp = []

    temp, weak_ans = generate_weak_ans(question)

    if is_correct_ans(answer, weak_ans) == True:
        return temp

    i = 1
    while i <= max_iter:
        hints = generate_hint(question, weak_ans)

        temp, weak_ans = get_better_ans(question, hints, temp)
        
        if is_correct_ans(answer,  weak_ans) == True:
            return temp
        i += 1
    return temp

def progressive_hint(question, answer):
    def get_better_ans(question, hints, temp):
        hints_str = ", ".join(map(str, hints))
        query = f'''Here is a Math Question: {question}. 
<Hint>
Hint: the answer is near ({hints_str})
<Hint>

<Task>
Your task is to solve a series of math word problems by providing the final answer. 
- Show your answer step by step.
- Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.
<Task>

<Requirements>
- Please anser this question in the format: `We know the Answer Hints: $Hints$. With the Answer Hints: $Hints$, we will answer the question`.
- Please answer the question step by step
- Please refer the hint for reference
- The solution you provided should be short and clear
- Each setp should less than 50 words, and you should solve this problem in less than 10 steps
<Requirements>
'''
        if len(query) > 4000:
            temp.append({'role':'user', 'content':query[:4000]})
        else:
            temp.append({'role':'user', 'content':query})
        better_ans = send_request([{'role':'user', 'content':query}])
        temp.append({'role':'assistant', 'content':better_ans})
        return temp, better_ans
    
    # 设置最大迭代次数为3
    max_iter = 3
    temp = []
    hints = []

    temp, weak_ans = generate_weak_ans(question)

    if is_correct_ans(answer, weak_ans) == True:
        return temp
    
    llm_answer = extract_ans_from_response(weak_ans)
        
    if isinstance(llm_answer, str):
        try:
            llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)[0]
        except IndexError:
            llm_answer = -10000000  # 或者你可以设置一个默认值
        except Exception as e:
            print(f"发生错误: {e}")
    llm_answer = delete_extra_zero(llm_answer)

    hint = llm_answer
    hints.append(hint)
    i = 1
    while i <= max_iter:

        temp, weak_ans = get_better_ans(question, hints, temp)
        
        if is_correct_ans(answer,  weak_ans) == True:
            return temp
        
        i += 1
        llm_answer = extract_ans_from_response(weak_ans)
        if isinstance(llm_answer, str):
            try:
                llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)[0]
            except IndexError:
                llm_answer = -10000000  # 或者你可以设置一个默认值
            except Exception as e:
                print(f"发生错误: {e}")
        hint = llm_answer
        hints.append(hint)
    llm_answer = delete_extra_zero(llm_answer)

    return temp

def combine_method(question, answer):
    def generate_hint_msg(question, weak_answer):
        query = f'''Question: {question}
Now we have a weak answer: 

<weak answer>
{weak_answer}
<weak answer> 

You should generate some hints to improve the answer.

Your creteria should include:

<criterion>
- The final answer must include “#### [value]” format (e.g., “#### 500”).
- Every number used in the steps must be provided by the question. For example, you should check if any number in the weak answer can not be found in the question. The intermediate results calculated through the variables in the question are not counted.
- The answer must directly correspond to the question asked. For example, if the question asks the number of Apple, the answer should be answer the amout of Apple, not banana.
- Each step should have a logical explanation justifying its inclusion (e.g., formulas).
<criterion>

<requirements>
- The hint should be less than 50 words. 
- The hint should not give the answer directly, so avoid any calculation.
- You Should not return improved answer, or comments to the weak answer.
- The hint should be return in bullet points.
- Hint SHOULD NOT provide steps to solve the question.
<requirements>'''
        hints = send_request([{'role':'user', 'content':query}])
        if len(hints) > 3000:
            return hints[0:3000]
        return hints
    
    def get_better_ans(question, hints, hint_msg, temp):
        hints_str = ", ".join(map(str, hints))

        query = f'''Here is a Math Question: 
<Question>
{question}
<Question>

<Task>
Your task is to solve a series of math word problems by providing the final answer. 
- Show your answer step by step.
- Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.
<Task>

<Requirements>
- Please answer the question step by step
- Please refer the hint for solution
- The solution you provided should be short and clear
- Each setp should less than 50 words, and you should solve this problem in less than 10 steps
<Requirements>

Here are some hints for you to reference:

<Hint>
Hint: the answer is near ({hints_str})
{hint_msg}
<Hint>
'''
        if len(query) > 4000:
            temp.append({'role':'user', 'content':query[:4000]})
        else:
            temp.append({'role':'user', 'content':query})
        better_ans = send_request([{'role':'user', 'content':query}])
        temp.append({'role':'assistant', 'content':better_ans})
        return temp, better_ans
    
    # 设置最大迭代次数为3
    max_iter = 3
    temp = []
    hints = []

    temp, weak_ans = generate_weak_ans(question)

    if is_correct_ans(answer, weak_ans) == True:
        return temp
    
    llm_answer = extract_ans_from_response(weak_ans)

    hint_num = llm_answer

    hints.append(hint_num)

    hint_msg = generate_hint_msg(question, weak_ans)

    i = 1
    while i <= max_iter:

        temp, weak_ans = get_better_ans(question, hints, hint_msg, temp)
        
        if is_correct_ans(answer,  weak_ans) == True:
            return temp
        
        i += 1

        llm_answer = extract_ans_from_response(weak_ans)

        if isinstance(llm_answer, str):
            try:
                llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)[0]
            except IndexError:
                llm_answer = -10000000  # 或者你可以设置一个默认值
            except Exception as e:
                print(f"发生错误: {e}")

        llm_answer = delete_extra_zero(llm_answer)

        hint_num = llm_answer

        hint_msg = generate_hint_msg(question, weak_ans)

        hints.append(hint_num)
        
    return temp

filename = 'data/GSM8K/test.jsonl'
# log_file_path = "log.jsonl" # zero-shot
log_file_path = "./output/method_combine.jsonl"
data_list = load_json_objects(filename)

count = 0
total_time = 0
total_tokens = 0


for idx, input in enumerate(data_list):
    # zero shot:
    # temp = zero_shot_prompt # zero-shot
    # temp[1]['content']=input["question"] # zero-shot

    # few shot:
    # temp = few_shot_prompt # few-shot
    # temp[-1]['content']=input["question"] # few-shot

    start_time = time.time()  # Start timing

    # res = send_request(temp)

    
    test_answer = extract_ans_from_response(input["answer"]) # ans from dataset
    if isinstance(test_answer, str):
        test_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', test_answer)[0]
    test_answer = delete_extra_zero(test_answer)

    # self-refine
    # llm_ans = self_refine(input["question"], input["answer"])
    # res = llm_ans[-1]["content"]

    # progressive-hint
    # llm_ans = progressive_hint(input["question"], input["answer"])
    # res = llm_ans[-1]["content"]

    # method combine
    llm_ans = combine_method(input["question"], input["answer"])
    res = llm_ans[-1]["content"]

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    total_time += elapsed_time  # Add to total time

    for message in llm_ans:
        if message['role'] == 'assistant':
            total_tokens += len(message['content'].split())
    
    avg_token = total_tokens/idx if idx > 0 else 0

    # compare answer
    llm_answer = extract_ans_from_response(res)
    if isinstance(llm_answer, str):
        try:
            llm_answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', llm_answer)[0]
        except IndexError:
            llm_answer = -10000000  # 或者你可以设置一个默认值
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
        "prompt": llm_ans, 
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
avg_token = total_tokens / len(data_list) if len(data_list) > 0 else 0

print(f"""Evaluating method combine:
Accuracy: {accuracy_rate} ({count}/1319)
Total inference time (seconds): {total_time}
Average token count per question: {avg_token}
""")