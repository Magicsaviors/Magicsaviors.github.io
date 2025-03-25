# 基于Qwen2.5-0.5B-Instruct的Lora微调

## 一、创建虚拟环境及环境配置

### 创建名为“Lora”的虚拟环境

```python
conda create -n Lora python=3.11
```

### 本地模型部署
这里借助[ModelScope](https://modelscope.cn/my/overview)社区进行模型的下载，推荐用```git bash```clone到本地

```python
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
```

### 安装第三方库

```python
pip install modelscope==1.18.0
pip install transformers==4.44.2
pip install sentencepiece==0.2.0
pip install accelerate==0.34.2
pip install datasets==2.20.0
pip install peft==0.11.1
```

### 模型加载

```python
# 加载预训练的语言模型，自动适配数据类型并映射到指定设备
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 加载与模型对应的分词器
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct")
```
## 二、指令集的构建

### 构建目的

通过构建包含 “**​任务描述**（Instruction） + **输入**（Input） + **输出**（Output）”​ 的三元组数据集，让模型学习如何根据指令完成任务。增强模型的​任务泛化能力，使其能处理未见过的任务类型（如进行翻译、摘要任务）。

### 构建示例

```python
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
```

其中，```instruction``` 是用户指令，告知模型其需要完成的任务；```input``` 是用户输入，是完成用户指令所必须的输入内容；```output``` 是模型应该给出的输出。

我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。本文以```chat-黛玉```为例构建有关林黛玉的对话指令集，示例如下：

```python
{
  "instruction": "窗前的竹子可要修剪？",
  "input": "",
  "output": "留着听雨声罢，李义山说的'留得残荷听雨声'倒是知己。" 
}
```

## 三、数据预处理

```python
# 训练数据预处理方法
def preprocess(tokenizer, batch_messages):
    input_list = []
    target_list = []
    
    im_start = tokenizer('<|im_start|>').input_ids
    im_end = tokenizer('<|im_end|>').input_ids
    newline = tokenizer('\n').input_ids
    pad_token = tokenizer.pad_token_id  # 获取 pad token 的 ID
    ignore = [-100]
    
    for group in batch_messages:
        role = tokenizer(f"user\n{group['instruction']}").input_ids
        content = tokenizer(f"assistant\n{group['output']}").input_ids
        
        input_ids = im_start + role + im_end + newline
        target_ids = im_start + ignore * len(role) + content + im_end + newline
        
        input_list.append(input_ids)
        target_list.append(target_ids)
    
    # 计算最大长度，并确保所有序列具有相同长度
    max_len = max([len(ids) for ids in input_list])
    
    padded_input_list = []
    padded_target_list = []
    
    for input_ids, target_ids in zip(input_list, target_list):
        # 确保精确填充到 max_len
        padded_input = input_ids[:max_len] if len(input_ids) > max_len else input_ids + [pad_token] * (max_len - len(input_ids))
        padded_target = target_ids[:max_len] if len(target_ids) > max_len else target_ids + ignore * (max_len - len(target_ids))
        
        padded_input_list.append(padded_input)
        padded_target_list.append(padded_target)
    
    batch_input_ids = torch.tensor(padded_input_list, dtype=torch.long)
    batch_target_ids = torch.tensor(padded_target_list, dtype=torch.long)
    
    # 明确设置 attention_mask
    batch_mask = (batch_input_ids != pad_token).long()
    
    return batch_input_ids, batch_target_ids, batch_mask
```

接下来我们逐步拆解一下预处理流程：

### （一）定义与处理函数

函数接收**分词器**(tokenizer)和**批量消息**(batch_messages)作为输入参数。```input_list```用于存储编码后的输入序列，```target_list```存储目标序列.

```python
def preprocess(tokenizer, batch_messages):
    input_list = []
    target_list = []
```

### （二）特殊标记初始化

- ```<|im_start|>```和```<|im_end|>```标记对话的开始与结束，用于结构化对话数据
- ```\n```用于分隔角色与内容，增强模型对对话结构的理解
- ```pad_token```作为填充符(PAD)，统一序列长度
- ```ignore=[-100]```表示在计算损失时忽略对应位置的预测值


```python
    im_start=tokenizer('<|im_start|>').input_ids
    im_end=tokenizer('<|im_end|>').input_ids
    newline=tokenizer('\n').input_ids
    pad_token = tokenizer.pad_token_id  # 获取 pad token 的 ID
    ignore=[-100]
```

### （三）指令集模板化输入

```python
<|im_start|>user
{用户指令+输入}<|im_end|>
<|im_start|>assistant
```
我们依照Qwen的指令模板进行如下构建：
```python
    for group in batch_messages:
        role = tokenizer(f"user\n{group['instruction']}").input_ids
        content = tokenizer(f"assistant\n{group['output']}").input_ids
        
        input_ids = im_start + role + im_end + newline
        target_ids = im_start + ignore * len(role) + content + im_end + newline
        
        input_list.append(input_ids)
        target_list.append(target_ids)
```

###  （四）序列填充对齐

```python
 # 计算最大长度，并确保所有序列具有相同长度
    max_len = max([len(ids) for ids in input_list])
    
    padded_input_list = []
    padded_target_list = []
    
    for input_ids, target_ids in zip(input_list, target_list):
        # 确保精确填充到 max_len
        padded_input = input_ids[:max_len] if len(input_ids) > max_len else input_ids + [pad_token] * (max_len - len(input_ids))
        padded_target = target_ids[:max_len] if len(target_ids) > max_len else target_ids + ignore * (max_len - len(target_ids))
        
        padded_input_list.append(padded_input)
        padded_target_list.append(padded_target)
    
    batch_input_ids = torch.tensor(padded_input_list, dtype=torch.long)
    batch_target_ids = torch.tensor(padded_target_list, dtype=torch.long)
    
    # 明确设置 attention_mask
    batch_mask = (batch_input_ids != pad_token).long()  #标识有效内容位置（标注哪些是真实题目，哪些是填充的空白）
    
    return batch_input_ids, batch_target_ids, batch_mask
```

### （五）Lora参数配置

```lora_alpha```：控制低秩矩阵对原始权重的调整幅度，与学习率共同影响参数更新幅度，最终权重调整量为$(loRA_B × loRA_A) × \frac{\alpha}{r}$

```python
from peft import LoraConfig, get_peft_model  

# 配置 LoRA  
lora_config = LoraConfig(  
    r=8,  # LoRA的秩，影响参数量  
    lora_alpha=32,  # LoRA alpha系数  
    lora_dropout=0.1,  # Dropout率，用于正则化,避免过拟合，增强模型泛化能力
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 模块名称  
    task_type="CAUSAL_LM",  # 任务类型,本处用于文本生成、对话
)  

# 将 LoRA 应用到 Qwen 模型  
model = get_peft_model(model, lora_config)  

# 查看可训练参数  
model.print_trainable_parameters()
```

#### Qwen模型LoRA目标模块说明

| 模块名称       | 所属结构          | 功能描述                           | 参数影响度 | 推荐指数 |
|----------------|-------------------|-----------------------------------|-----------|----------|
| `q_proj`       | 自注意力层        | 生成查询向量(Query)               | ★★★★★     | ⭐⭐⭐⭐    |
| `k_proj`       | 自注意力层        | 生成键向量(Key)                   | ★★★★☆     | ⭐⭐⭐     |
| `v_proj`       | 自注意力层        | 生成值向量(Value)                 | ★★★★★     | ⭐⭐⭐⭐    |
| `o_proj`       | 自注意力层        | 输出投影(Output projection)       | ★★★★☆     | ⭐⭐⭐     |
| `gate_proj`    | FFN层             | 门控线性变换(Gated linear)        | ★★★☆☆     | ⭐⭐      |
| `up_proj`      | FFN层             | 升维投影(Up projection)           | ★★★★☆     | ⭐⭐⭐     |
| `down_proj`    | FFN层             | 降维投影(Down projection)         | ★★★★☆     | ⭐⭐⭐     |

### （六）训练模型

```python
model.train()

for i in range(100):
    print("第{}轮训练".format(i))
    batch_input_ids,batch_target_ids,batch_mask=preprocess(tokenizer,datas)
    model_outputs=model(batch_input_ids.to(device)) #输出包含 logits：模型预测的概率分布

    output_tokens=model_outputs.logits.argmax(dim=-1) #argmax(dim=-1)：取概率最大的token

    logits=model_outputs.logits[:,:-1,:]  # 去掉最后一个预测
    targets=batch_target_ids[:,1:].to(device)  # 去掉第一个token
    print("训练了第{}次".format(i))
    print('logits:',logits.shape) # 模型输出
    print('targets:',targets.shape) # 拟合目标

    from torch.nn import CrossEntropyLoss

    # 损失
    loss_fn=CrossEntropyLoss()
    loss=loss_fn(logits.reshape(-1,logits.size(2)),targets.reshape(-1)) 
    print('loss:',loss)

    # 优化器
    optimizer=torch.optim.SGD(model.parameters())
    optimizer.zero_grad()

    # 求梯度
    loss.backward()

    # 梯度下降
    optimizer.step()
```

### （七）准备对话

#### 构建对话函数

```python
def chat(prompt):
    # 预设角色背景
    system_prompt = "你是林黛玉，贾宝玉的表妹。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 生成配置
    generation_config = {
        "max_new_tokens": 128,  # 控制输出长度
        "temperature": 0.7,     # 控制随机性
        "top_p": 0.9,           # 核采样
        "repetition_penalty": 1.2  # 降低重复
    }
    
    # 模型交互
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        **generation_config
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

```

#### 输入指令

```python
prompt="这是暹罗进贡的茶叶"
chat(prompt)
```