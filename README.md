# CodeMind

## 소개
`CodeMind-gemma-2b`는 코딩 테스트 문제 해결 및 프로그래밍 교육을 지원하는 언어 모델입니다. 이 모델은 LeetCode 사용자 제출물과 관련 유튜브 캡션을 활용하여 문제 해결에 대한 설명과 코드 예제를 제공합니다.

## 모델 세부 정보
- **모델 이름**: CodeMind
- **기본 모델**: google/gemma-2b
- **언어**: 영어
- **모델 크기**: 2.51B 파라미터
- **라이선스**: MIT

## 주요 기능
- 코딩 테스트 문제 해결
- 프로그래밍 개념 설명
- 관련 코드 스니펫 생성

## 훈련 데이터
- **LeetCode 사용자 제출물**: 다양한 알고리즘 문제의 파이썬 솔루션
- **유튜브 캡션**: LeetCode 문제에 대한 설명 및 단계별 가이드

## 사용된 라이브러리
- [transformers](https://github.com/huggingface/transformers): 자연어 처리 모델을 위한 라이브러리
- [datasets](https://github.com/huggingface/datasets): 데이터셋 처리 및 관리 라이브러리
- [bitsandbytes](https://github.com/facebookresearch/bitsandbytes): 최적화된 연산을 위한 라이브러리
- [peft](https://github.com/peft/peft): 파인 튜닝을 위한 라이브러리
- [trl](https://github.com/huggingface/trl): 언어 모델 튜닝을 위한 라이브러리
- [pandas](https://pandas.pydata.org/): 데이터 조작을 위한 라이브러리
- [torch](https://pytorch.org/): PyTorch, 딥러닝 프레임워크

## 파일 구조
- **dataset/**: 데이터셋 파일을 포함합니다.
- **eval/**: 평가 스크립트를 포함합니다.
- **fine-tuning/**: 미세 조정 관련 노트북 및 스크립트를 포함합니다.
  - `gemma-1.1-2b-it peft qlora.ipynb`: 미세 조정 과정에 대한 세부 사항이 포함된 노트북입니다.
- **.gitignore**: Git 무시 규칙이 정의된 파일입니다.
- **LICENSE**: MIT 라이선스 파일입니다.
- **README.md**: 프로젝트에 대한 설명 파일입니다.
- **demo.ipynb**: 데모 노트북으로 모델 사용 예제가 포함되어 있습니다.
- **requirements.txt**: 프로젝트 의존성 목록이 포함되어 있습니다.
- **utils.py**: 유틸리티 함수들이 포함되어 있습니다.

## 사용 방법
이 모델은 HuggingFace의 모델 허브를 통해 액세스할 수 있으며, API를 사용하여 응용 프로그램에 통합할 수 있습니다. 코딩 문제 또는 프로그래밍 관련 질문을 제공하면 모델이 관련 설명, 코드 스니펫 또는 가이드를 생성합니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("kreimben/CodeMind-gemma-2b")
model = AutoModelForCausalLM.from_pretrained("kreimben/CodeMind-gemma-2b")

inputs = tokenizer("코딩 문제나 질문을 여기에 입력하세요", return_tensors="pt")
outputs = model.generate(inputs.input_ids)
print(tokenizer.decode(outputs[0]))
```

## 훈련 과정

### 모델 및 토크나이저 로드
```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = 'google/gemma-1.1-2b-it'
token = os.getenv('HF_READ')

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, token=token)
model.config.use_cache = False
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
```

### LoRA 구성 및 모델 준비
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

model = prepare_model_for_kbit_training(model)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 데이터 준비
```python
import pandas as pd
from datasets import Dataset

submission_dataset = datasets.load_dataset('kreimben/leetcode_user_submissions_only_python', split='train').to_pandas()
submission_dataset = submission_dataset[['title', 'question_hints', 'question_content', 'content']]
captions_dataset = datasets.load_dataset('kreimben/leetcode_with_youtube_captions', split='train').to_pandas()
captions_dataset = captions_dataset[['title', 'question_hints', 'question_content', 'cc_content']]
captions_dataset.rename(columns={'cc_content': 'content'}, inplace=True)

dataset = pd.concat([submission_dataset, captions_dataset])
del submission_dataset, captions_dataset

dataset = Dataset.from_pandas(dataset)
GEMMA_2B_IT_MODEL_PREFIX_TEXT = "Below is an coding test problem. Solve the question."

def generate_prompt(data_point):
    return f"<bos><start_of_turn>user {GEMMA_2B_IT_MODEL_PREFIX_TEXT}

I don't know {data_point['title']} problem. give me the insight or appoach.

this is problem's hint.
{data_point['question_hints']}

here are some content of question.
{data_point['question_content']}<end_of_turn>
<start_of_turn>model {data_point['content']}<end_of_turn><eos>"

text_column = [generate_prompt(data_point) for data_point in dataset]
dataset = dataset.add_column("prompt", text_column)
```

### 훈련
```python
from trl import SFTTrainer
import transformers
import torch

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt",
    peft_config=lora_config,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=transformers.TrainingArguments(
        output_dir='out',
        bf16=True,
        max_steps=100,
        warmup_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        logging_steps=20,
        report_to='wandb',
    ),
)

trainer.train()
```

## 평가
모델의 성능은 다음과 같이 평가되었습니다:

| Metric       | Value  |
|--------------|--------|
| Average      | 41.62  |
| ARC          | 41.81  |
| HellaSwag    | 59.03  |
| MMLU         | 37.26  |
| TruthfulQA   | 43.45  |
| Winogrande   | 59.91  |
| GSM8K        | 8.26   |

## 제한 사항 및 윤리적 고려사항
- 모델의 출력은 학습 데이터에 기반하므로 항상 정확하지 않을 수 있습니다.
- 중요한 결정이나 실세계 문제 해결에 모델 출력을 사용하기 전에 반드시 검증이 필요합니다.
