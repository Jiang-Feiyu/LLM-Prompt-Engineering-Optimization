import re

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



if __name__ == '__main__':
    test_solution = "First find how many gigabytes are in 40% of the file: 200 GB * 40% = <<200*40*.01=80>>80 GB\nThen divide that number by the download rate to find the time until Windows restarts: 80 GB / 2 GB/minute = <<80/2=40>>40 minutes\nThen find the time to download the whole file after the restart: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, to download the whole file, and to wait for Windows to update: 40 minutes + 100 minutes + 20 minutes = <<40+100+20=160>>160 minutes\n#### 160"
    answer = extract_ans_from_response(test_solution)
    if isinstance(answer, str):
        answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', answer)[0]
    answer = delete_extra_zero(answer)
    print(answer)