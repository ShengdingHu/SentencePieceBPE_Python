corpus = [
    "This is the Hugging Face Course.\n",
    "This chapter is about token token token tokenization.\n",
    "This section shows several tokenizer algorithms.\n",
    "Hopefully, you will be able to understand how they are trained and generate tokens.\n",
]

import random

class TokenInstance:
    def __init__(self, content, prev_token=None, next_token=None):
        self.content = content
        self.prev_token = prev_token
        self.next_token = next_token
        self.is_valid = True
        self.hash = str(random.randint(0,100000000))

    def __repr__(self,):
        return   "({}, {}, {})".format(self.prev_token.content if self.prev_token is not None else '',
                                     self.content, 
                                     self.next_token.content if self.next_token is not None else '')

corpus_tokens = []
token_objs = {}
for sidx, sentence in enumerate(corpus):
    sentence_tokens = []
    prev_token = None
    for cidx, character in enumerate(sentence):
        cur_token = TokenInstance(content=character, prev_token=prev_token)
        if prev_token is not None:
            prev_token.next_token = cur_token
        sentence_tokens.append(cur_token)
        prev_token = cur_token
        token_objs[cur_token.hash] = cur_token

    corpus_tokens.append(sentence_tokens)

class BigramInfo:
    def __init__(self,):
        self.freq = 0
        self.appear = {}
    def __repr__(self,):
        return f"freq: {self.freq}"

token_info_dict = {}
bigram_info_dict = {}
for sidx, sentence in enumerate(corpus_tokens):
    for cidx, token in enumerate(sentence):
        if cidx + 1 < len(sentence): # 如果不是最后一个token
            bigram = (sentence[cidx].content, sentence[cidx+1].content)
            
            if bigram not in bigram_info_dict:
                bigram_info_dict[bigram] = BigramInfo()
            bigram_info_dict[bigram].freq += 1
            bigram_info_dict[bigram].appear[(sentence[cidx], sentence[cidx+1])] = 1


def find_best_merge(bigram_info_dict):
    max_freq = 0
    best_bigram = None
    for bigram in bigram_info_dict:
        if bigram_info_dict[bigram].freq > max_freq:
            best_bigram = bigram
            max_freq = bigram_info_dict[bigram].freq
    return best_bigram
    
def merge_tokens(bigram,  bigram_info_dict):
    # 找到这个bigram涉及到的位置，逐个处理
    appears = [a for a in bigram_info_dict[bigram].appear.keys()]
    for appear in appears:
        # 对于每个该bigram出现的位置，需要做三件事
        # 1. 形成一个新的TokenInstance
        left_token, right_token = appear
        if not(left_token.is_valid and right_token.is_valid):
            continue # (如果新的TokenInstance中有位置已经被用过了, 例如a a a a合并aa)
        left_token.is_valid = False
        right_token.is_valid = False
        new_token = TokenInstance(content="".join([left_token.content, right_token.content]),
                             prev_token=left_token.prev_token,
                             next_token=right_token.next_token)
        token_objs[new_token.hash] = new_token
        
        # 2. 将这个新的token, 和左边、右边的token联合起来变成新的bigram
        # 2.1 左边
        if new_token.prev_token is not None:
            # 从appear里移掉old_bigram, old_bigram的频率减一
            old_bigram = (new_token.prev_token.content, left_token.content)
            bigram_info_dict[old_bigram].freq -= 1
            bigram_info_dict[old_bigram].appear.pop((new_token.prev_token, left_token))
            # 在bigram表里加上新的bigram, 记录出现的位置
            new_bigram =  (new_token.prev_token.content, new_token.content)
            if new_bigram not in bigram_info_dict:
                bigram_info_dict[new_bigram] = BigramInfo()
            bigram_info_dict[new_bigram].freq += 1
            bigram_info_dict[new_bigram].appear[(new_token.prev_token, new_token)] = 1
        
        # 2.1 右边
        if new_token.next_token is not None:
            # 从appear里移掉old_bigram, old_bigram的频率减一
            old_bigram = (right_token.content, new_token.next_token.content)
            bigram_info_dict[old_bigram].freq -= 1
            bigram_info_dict[old_bigram].appear.pop((right_token, new_token.next_token))
            # 在bigram表里加上新的bigram, 记录出现的位置
            new_bigram =  (new_token.content, new_token.next_token.content)
            if new_bigram not in bigram_info_dict:
                bigram_info_dict[new_bigram] = BigramInfo()
            bigram_info_dict[new_bigram].freq += 1
            bigram_info_dict[new_bigram].appear[(new_token, new_token.next_token)] = 1
        
        # 把前后两个token链接到新token上
        if new_token.prev_token is not None:
            new_token.prev_token.next_token = new_token
        if new_token.next_token is not None:
            new_token.next_token.prev_token = new_token

    # 最后把原来的bigram删除掉
    bigram_info_dict.pop(bigram)

        
num_merges = 150  # 设置目标词汇表大小
merge_rules = []  # 初始化空字典来存储合并
while len(merge_rules) < num_merges:  # 当词汇表大小未达到目标时
    best_bigram = find_best_merge(bigram_info_dict)  # 找到最佳合并
    if best_bigram is None: # 已经没有可以合并的了
        break
    splits = merge_tokens(best_bigram, bigram_info_dict)  # 合并字符对
    merge_rules.append((best_bigram, [''.join(best_bigram)]) )# 更新合并规则




# 定义分词函数
def tokenize_slow(text, merge_rules):
    split = list(text)
    # 遍历预定义的合并规则
    for pair, merge in merge_rules:
        print(split)
        # 遍历所有单词的字符列表
        i = 0  # 初始化索引
        # 遍历当前单词的字符列表
        while i < len(split) - 1:
            # 如果找到匹配的字符对
            if split[i] == pair[0] and split[i + 1] == pair[1]:
                # 使用合并规则合并字符对
                split = split[:i] + merge + split[i + 2 :]
            else:
                i += 1  # 如果未找到匹配的字符对，增加索引
    # 将所有单词的字符列表合并为一个大的字符列表，并返回
    return split

print(merge_rules)
tokenized_sentence = tokenize_slow("tokenization into tokens is fast", merge_rules=merge_rules)
print(tokenized_sentence)
