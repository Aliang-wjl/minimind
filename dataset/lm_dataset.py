from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """把 DPO 偏好样本转换为 chosen/rejected 两条可比较的 token 序列。

    每条 JSONL 样本应包含：
    ``chosen``：共享对话上下文 + 更优的 assistant 回答；
    ``rejected``：相同上下文 + 较差的 assistant 回答。

    本类会为两条分支分别生成：
    - ``x``：去掉最后一个 token 的模型输入；
    - ``y``：去掉第一个 token 的 next-token 目标；
    - ``mask``：只在 assistant 回答区域为 1，其余上下文和 padding 为 0。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        """加载偏好数据，并预先确定 assistant 区间的起止 token 模式。"""
        # 初始化 PyTorch Dataset 基类。
        super().__init__()
        # 保存 tokenizer，用于 chat template、编码和特殊 token 查询。
        self.tokenizer = tokenizer
        # chosen/rejected 都会截断或补齐到同一最大长度。
        self.max_length = max_length
        # tokenizer 没有显式 pad_token_id 时退回 0，保证 padding 操作可用。
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 编码“assistant 消息开始”标记，之后用它在完整 token 序列中定位回答起点。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 编码“一条消息结束”标记，用于定位 assistant 回答终点。
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        # Hugging Face datasets 以流畅的列式结构加载 JSONL 的全部偏好样本。
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        """返回偏好对数量；一个样本同时包含一条 chosen 和一条 rejected。"""
        return len(self.samples)

    def __getitem__(self, index):
        """读取一个偏好对并构造 DPO 训练所需的六个张量。"""
        # 取得第 index 条字典样本。
        sample = self.samples[index]
        # chosen 是若干 {role, content} 组成的消息列表，末尾为更优回答。
        chosen = sample['chosen']
        # rejected 通常具有相同上下文，但末尾 assistant 回答质量更差。
        rejected = sample['rejected']

        # 用项目 tokenizer 的 chat template 将 chosen 多轮消息序列化成完整文本。
        chosen_prompt = self.tokenizer.apply_chat_template(
            # 不在这里直接分词，且数据中已有 assistant 回答，所以不追加生成提示。
            chosen, tokenize=False, add_generation_prompt=False
        )

        # 对 rejected 使用完全相同的模板，确保两分支只在回答内容上产生差异。
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 以同一个随机决策处理 chosen/rejected 的空 <think> 标签，避免两支模板格式不一致。
        remove_empty_think = random.random() > 0.2
        # 按上述共同决策处理优选回答文本。
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        # 较差回答执行同样处理，使偏好比较不被随机模板差异污染。
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)

        # chosen 编码后截断到 max_length；不足部分右侧补 pad，以便组成固定形状 batch。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        # rejected 使用同样的长度规则。
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 取得 chosen 的完整 token ID 列表，长度为 max_length。
        chosen_input_ids = chosen_encoding['input_ids']
        # 标记 chosen 中哪些 assistant token 应计入序列对数概率。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        # 取得 rejected 的完整 token ID 列表。
        rejected_input_ids = rejected_encoding['input_ids']
        # 独立定位 rejected 的 assistant 回答区域。
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # x 去掉最后一个 token，作为“根据当前位置及之前内容预测下一个 token”的输入。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        # y 去掉第一个 token，与 x 的每个位置错开一位形成 next-token 目标。
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        # mask 也右移一位，与 y 而不是 x 对齐。
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        # rejected 分支执行完全相同的移位操作。
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # DataLoader 会把相同键的张量堆叠成 [B,T]；训练脚本再拼成 [2B,T]。
        return {
            # 优选分支的模型输入。
            'x_chosen': x_chosen,
            # 优选分支的预测目标。
            'y_chosen': y_chosen,
            # 优选分支中需要累加 log-prob 的 assistant 位置。
            'mask_chosen': mask_chosen,
            # 拒绝分支的模型输入。
            'x_rejected': x_rejected,
            # 拒绝分支的预测目标。
            'y_rejected': y_rejected,
            # 拒绝分支中需要累加 log-prob 的 assistant 位置。
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """在 token 序列中定位 assistant 回答，并生成 0/1 loss mask。

        system/user 上下文虽然会作为模型输入参与注意力计算，但 mask 为 0，不进入
        DPO 的回答序列 log-prob；assistant 内容及其结束标记的 mask 为 1。
        """
        # 初始全部忽略，包括 system、user、padding 以及尚未识别出的 assistant token。
        loss_mask = [0] * len(input_ids)
        # i 是在完整序列中向右扫描的游标。
        i = 0

        # 逐位置搜索 assistant 开始标记，支持一条样本中存在多轮 assistant 回复。
        while i < len(input_ids):
            # 当前切片与 bos_id 完全相同时，说明发现了一段 assistant 消息。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 回答正文从 assistant 头标记之后开始，头标记自身不计入偏好概率。
                start = i + len(self.bos_id)
                # end 从正文起点开始寻找消息结束标记。
                end = start

                # 未到序列末尾前持续向右扫描。
                while end < len(input_ids):
                    # 找到 eos_id 后停止；此时 end 指向结束标记开头。
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    # 当前不是结束标记，继续检查下一个 token 位置。
                    end += 1

                # 将回答正文连同 EOS 标记设为 1，但绝不越过 max_length。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                # 找到 EOS 时直接跳过整段回答；未找到时结束整个扫描。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 当前不是 assistant 起点，游标前进一步。
                i += 1

        # 返回与完整 input_ids 等长的 0/1 列表，调用者随后会与 y 一起右移。
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
