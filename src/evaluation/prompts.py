
def format_kobest_boolq(sample):
    return f"지문: {sample['paragraph']}\n질문: {sample['question']}\n위 질문에 대한 답이 참(True)이면 1, 거짓(False)이면 0을 선택하세요.\n정답:"

def format_kobest_copa(sample):
    return f"전제: {sample['premise']}\n질문: {sample['question']}\n선택지 1: {sample['alternative_1']}\n선택지 2: {sample['alternative_2']}\n정답(1 또는 2):"

def format_kobest_hellaswag(sample):
    return f"문맥: {sample['context']}\n이어질 문장으로 가장 적절한 것은?\n1: {sample['ending_1']}\n2: {sample['ending_2']}\n3: {sample['ending_3']}\n4: {sample['ending_4']}\n정답(1-4):"

def format_kobest_sentineg(sample):
    return f"문장: {sample['sentence']}\n위 문장의 긍정/부정 여부를 판단하세요. (긍정: 1, 부정: 0)\n정답:"

def format_kobest_wic(sample):
    return f"단어: {sample['word']}\n문장 1: {sample['context_1']}\n문장 2: {sample['context_2']}\n위 두 문장에서 해당 단어가 같은 의미로 쓰였으면 1, 다르면 0을 선택하세요.\n정답:"

def format_kmmlu(sample):
    options = f"A. {sample['A']}\nB. {sample['B']}\nC. {sample['C']}\nD. {sample['D']}"
    return f"주제: {sample['Category']}\n질문: {sample['question']}\n{options}\n정답(A-D):"

def format_haerae(sample):
    # HAE-RAE structure varies, but generally has query and options
    # Assuming standard multiple choice if options are present
    # Currently we need to inspect specific task structure more closely for universal formatter
    # But often: 'query', 'options' key
    # Based on inspection, let's look at what we load.
    # LogicKor is different.
    # For now, a generic formatter if feasible, or task specific.
    # The user didn't give strict requirements on prompt engineering, so minimal viable prompts are okay.
    return f"질문: {sample.get('query', '')}\n{sample.get('options', '')}\n정답:" # Simplified

def format_logickor(sample):
    # LogicKor is a list of multi-turn questions
    # But usually we format the first turn for simple eval or handle chat structure.
    # LogicKor usually has 'questions' as a list of strings.
    questions = sample['questions']
    if isinstance(questions, list) and len(questions) > 0:
        return questions[0] # Evaluate single turn for now or pass context
    return str(questions)
