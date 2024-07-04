import torch
from transformers import AutoConfig, BloomTokenizerFast, BloomForSequenceClassification
import argparse


def setup_reward_model(args):
    # 加载配置
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=1)

    # 使用 BloomTokenizerFast
    tokenizer = BloomTokenizerFast.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    # 使用 BloomForSequenceClassification
    model = BloomForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    if args.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    return tokenizer, model


def test_reward_model(tokenizer, model, args):
    # 准备输入数据
    question = "高血压患者的日常饮食应该注意什么?"
    answer_chosen = "高血压患者应该注意低盐饮食,多吃水果蔬菜,避免高脂肪食物。"
    answer_rejected = "高血压患者的饮食没有特别要注意的,想吃什么就吃什么。"

    # 编码输入
    inputs_chosen = tokenizer(question, answer_chosen, return_tensors="pt", max_length=512, truncation=True,
                              padding="max_length")
    inputs_rejected = tokenizer(question, answer_rejected, return_tensors="pt", max_length=512, truncation=True,
                                padding="max_length")

    if args.use_gpu and torch.cuda.is_available():
        inputs_chosen = {k: v.cuda() for k, v in inputs_chosen.items()}
        inputs_rejected = {k: v.cuda() for k, v in inputs_rejected.items()}

    print("Chosen input shape:", inputs_chosen['input_ids'].shape)
    print("Rejected input shape:", inputs_rejected['input_ids'].shape)

    # 模型前向传播
    with torch.no_grad():
        outputs_chosen = model(**inputs_chosen)
        outputs_rejected = model(**inputs_rejected)

    print("Chosen output shape:", outputs_chosen.logits.shape)
    print("Rejected output shape:", outputs_rejected.logits.shape)
    print("Chosen score:", outputs_chosen.logits.item())
    print("Rejected score:", outputs_rejected.logits.item())

    # 计算奖励模型的损失
    loss = -torch.nn.functional.logsigmoid(outputs_chosen.logits - outputs_rejected.logits).mean()
    print("Loss:", loss.item())

    # 打印设备信息
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {inputs_chosen['input_ids'].device}")
    print(f"Output device: {outputs_chosen.logits.device}")


def main():
    parser = argparse.ArgumentParser(description="Test Reward Model")
    parser.add_argument("--model_name_or_path", type=str, default="merged-sft", help="Path to pretrained model")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")

    args = parser.parse_args()

    print("Parsed arguments:", args)

    tokenizer, model = setup_reward_model(args)
    test_reward_model(tokenizer, model, args)


if __name__ == "__main__":
    main()