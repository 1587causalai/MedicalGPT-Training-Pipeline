import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoModel, AutoConfig

print("脚本开始执行...")


def find_all_linear_names(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'lm_head' not in name and 'score' not in name:
                names = name.split('.')
                linear_layers.append(names[0] if len(names) == 1 else names[-1])
    return sorted(set(linear_layers))


class DiscoRewardModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_labels=1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.mean_head = nn.Linear(self.config.hidden_size, num_labels)
        self.log_var_head = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0]
        mean = self.mean_head(pooled_output)
        log_var = self.log_var_head(pooled_output)
        return (mean, log_var)

    def estimate_preference(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        mean_1, log_var_1 = self.forward(input_ids_1, attention_mask_1)
        mean_2, log_var_2 = self.forward(input_ids_2, attention_mask_2)

        var_1 = torch.exp(log_var_1)
        var_2 = torch.exp(log_var_2)

        # 计算偏好概率
        pref_prob = torch.special.ndtr((mean_1 - mean_2) / torch.sqrt(var_1 + var_2))

        return pref_prob

print("\n正在加载模型和tokenizer...")
model_name = "merged-sft"
config = AutoConfig.from_pretrained(model_name, num_labels=1)
model = DiscoRewardModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"模型名称: {model_name}")


target_modules = find_all_linear_names(model)
print(f"\n找到的可训练线性层: {target_modules}")

print("\n正在配置PEFT(LoRA)...")
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_modules
)
print(f"PEFT配置: {peft_config}")
model = get_peft_model(model, peft_config)
print("\nPEFT模型创建成功")

print("\n正在加载数据集...")
dataset = load_dataset("json", data_files={"train": "./data/reward/dpo_zh_500.jsonl"})
print(f"数据集信息: {dataset}")

# 创建一个小的评估数据集
eval_dataset = dataset["train"].select(range(10))  # 只选择前10个样本用于评估


def preprocess_function(examples):
    chosen = [
        f"{examples['system'][i]}\n\nHuman: {examples['question'][i]}\n\nAssistant: {examples['response_chosen'][i]}"
        for i in range(len(examples['question']))]
    rejected = [
        f"{examples['system'][i]}\n\nHuman: {examples['question'][i]}\n\nAssistant: {examples['response_rejected'][i]}"
        for i in range(len(examples['question']))]

    tokenized_chosen = tokenizer(chosen, truncation=True, padding="max_length", max_length=16)
    tokenized_rejected = tokenizer(rejected, truncation=True, padding="max_length", max_length=16)

    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


# 处理训练数据集
tokenized_train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 处理评估数据集
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print("\n正在设置训练参数...")
training_args = TrainingArguments(
    output_dir="outputs-rm-v1",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.05,
    weight_decay=0.001,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    remove_unused_columns=False,
    fp16=True,  # 启用混合精度训练
)
print(f"训练参数: {training_args}")


class RewardDataCollatorWithPadding:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = {
            "input_ids_chosen": torch.tensor([f["input_ids_chosen"] for f in features], dtype=torch.long),
            "attention_mask_chosen": torch.tensor([f["attention_mask_chosen"] for f in features], dtype=torch.long),
            "input_ids_rejected": torch.tensor([f["input_ids_rejected"] for f in features], dtype=torch.long),
            "attention_mask_rejected": torch.tensor([f["attention_mask_rejected"] for f in features], dtype=torch.long),
        }
        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        mean_chosen, log_var_chosen = model(input_ids=inputs["input_ids_chosen"],
                                            attention_mask=inputs["attention_mask_chosen"])
        mean_rejected, log_var_rejected = model(input_ids=inputs["input_ids_rejected"],
                                                attention_mask=inputs["attention_mask_rejected"])

        var_chosen = torch.exp(log_var_chosen)
        var_rejected = torch.exp(log_var_rejected)

        # 计算偏好概率，但不作为输出
        pref_prob = torch.special.ndtr((mean_chosen - mean_rejected) / torch.sqrt(var_chosen + var_rejected))
        loss = -torch.log(pref_prob).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": mean_chosen,  # 使用均值作为 "reward"
                "rewards_rejected": mean_rejected
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            mean_chosen, log_var_chosen = model(input_ids=inputs["input_ids_chosen"],
                                                attention_mask=inputs["attention_mask_chosen"])
            mean_rejected, log_var_rejected = model(input_ids=inputs["input_ids_rejected"],
                                                    attention_mask=inputs["attention_mask_rejected"])

            var_chosen = torch.exp(log_var_chosen)
            var_rejected = torch.exp(log_var_rejected)

            pref_prob = torch.special.ndtr((mean_chosen - mean_rejected) / torch.sqrt(var_chosen + var_rejected))
            loss = -torch.log(pref_prob).mean()

        # 确保所有返回的张量都被分离
        return (loss.detach(), None, None)

    def training_step(self, *args, **kwargs):
        if self.state.global_step % 100 == 0:  # 每100步清理一次
            torch.cuda.empty_cache()
        return super().training_step(*args, **kwargs)


print("\n正在初始化Trainer...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # 使用新的小型评估数据集
    tokenizer=tokenizer,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=16)
)

print("\n开始训练...")
trainer.train()

print("\n正在保存模型...")
trainer.save_model("final_model")

print("\n脚本执行完毕")
