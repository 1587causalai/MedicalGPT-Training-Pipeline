import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import argparse

def get_model_and_tokenizer(args):
    if args.model_type.lower() == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained(
            args.model_name_or_path,
            use_fast=args.use_fast_tokenizer,
        )
        model_class = BloomForSequenceClassification
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=args.use_fast_tokenizer,
        )
        model_class = AutoModelForSequenceClassification
    
    return tokenizer, model_class

def setup_reward_model(args):
    tokenizer, model_class = get_model_and_tokenizer(args)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=1)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(',') if args.target_modules != "all" else None
        )
        model = get_peft_model(model, peft_config)

    if args.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}%")

    return tokenizer, model

def test_reward_model(tokenizer, model, args):
    question = "高血压患者的日常饮食应该注意什么?"
    answer_chosen = "高血压患者应该注意低盐饮食,多吃水果蔬菜,避免高脂肪食物。"
    answer_rejected = "高血压患者的饮食没有特别要注意的,想吃什么就吃什么。"

    inputs_chosen = tokenizer(question, answer_chosen, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs_rejected = tokenizer(question, answer_rejected, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    if args.use_gpu and torch.cuda.is_available():
        inputs_chosen = {k: v.cuda() for k, v in inputs_chosen.items()}
        inputs_rejected = {k: v.cuda() for k, v in inputs_rejected.items()}

    print("Chosen input shape:", inputs_chosen['input_ids'].shape)
    print("Rejected input shape:", inputs_rejected['input_ids'].shape)
    
    with torch.no_grad():
        outputs_chosen = model(**inputs_chosen)
        outputs_rejected = model(**inputs_rejected)
    
    print("Chosen output shape:", outputs_chosen.logits.shape)
    print("Rejected output shape:", outputs_rejected.logits.shape)
    print("Chosen score:", outputs_chosen.logits.item())
    print("Rejected score:", outputs_rejected.logits.item())

    loss = -torch.nn.functional.logsigmoid(outputs_chosen.logits - outputs_rejected.logits).mean()
    print("Loss:", loss.item())

    # 打印设备信息
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {inputs_chosen['input_ids'].device}")
    print(f"Output device: {outputs_chosen.logits.device}")

def main():
    parser = argparse.ArgumentParser(description="Test Reward Model")
    parser.add_argument("--model_type", type=str, default="bloom", help="Type of the model")
    parser.add_argument("--model_name_or_path", type=str, default="merged-sft", help="Path to pretrained model")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer")
    parser.add_argument("--use_peft", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to use PEFT")
    parser.add_argument("--target_modules", type=str, default="all", help="Target modules for LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")

    args = parser.parse_args()

    print("Parsed arguments:", args)

    tokenizer, model = setup_reward_model(args)
    test_reward_model(tokenizer, model, args)

if __name__ == "__main__":
    main()