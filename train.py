import argparse
import json
import time

from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# MODEL_TYPE = "pythia-full"
MODEL_TYPE = "llama-lora"
PER_DEVICE_TRAIN_BATCH_SIZE = 2

parser = argparse.ArgumentParser(
    description="Run the training benchmark on a GPU instance."
)
parser.add_argument(
    "--hf-auth-token", type=str, required=True, help="Your Hugging Face auth token."
)
args = parser.parse_args()


assert torch.cuda.is_available()
# We only use one GPU per node.
device = "cuda:0"


def load_model_and_tokenizer(model_type):
    if model_type == "pythia-full":
        # Largest Pythia model that does not OOM on an A10G with PyTorch DDP
        model_name = "EleutherAI/pythia-1.4b"

        config = AutoConfig.from_pretrained(model_name)
        config.init_device = device

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # Expected by Hugging Face for decoder-only architectures
        tokenizer.padding_side = "left"

        return model, tokenizer
    elif model_type == "llama-lora":
        model_name = "meta-llama/Llama-2-7b-chat-hf"

        config = AutoConfig.from_pretrained(
            model_name, use_auth_token=args.hf_auth_token
        )
        config.init_device = device

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_auth_token=args.hf_auth_token,
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
            # QLoRA paper (https://arxiv.org/abs/2305.14314):
            #
            #   "LoRA on all linear transformer block layers are required to match
            #    full finetuning performance"
            target_modules=r"(.*self_attn\.*_proj|.*mlp\.gate_proj_proj|.*mlp\.up_proj|.*mlp\.down_proj)",
        )
        model = get_peft_model(model, peft_config)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=args.hf_auth_token
        )
        # Text generation will misbehave unless we use the pad_token_id from Llama's default generation_config.json (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json)
        tokenizer.pad_token_id = 0
        # Expected by Hugging Face for decoder-only architectures
        tokenizer.padding_side = "left"

        return model, tokenizer
    else:
        raise AssertionError(f'Unknown model type "{model_type}"')


dataset = json.loads(open("all_examples.json").read())
assert isinstance(dataset, list)
assert isinstance(dataset[0], str)


def main():
    dist.init_process_group(backend="nccl")

    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=False, sampler=sampler
    )
    torch.manual_seed(0)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Initialize Hugging Face model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_type=MODEL_TYPE)

    # Wrap the model in DistributedDataParallel
    ddp_model = DistributedDataParallel(model)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # Prevent node-to-node variations in model loading time from affecting our estimates of per-step timings
    dist.barrier()

    start_time = time.time()
    total_steps = 0

    for epoch in range(100):
        sampler.set_epoch(epoch)

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()

            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            labels = inputs["input_ids"].clone()

            # Forward pass
            outputs = ddp_model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_steps += 1
            print(
                f"Average step time {1000 * (time.time() - start_time) / float(total_steps):.2f} ms "
                + f"(Rank {rank}/{world_size-1} - Epoch {epoch} - Epoch Step {i} - Total Step {total_steps})"
            )


if __name__ == "__main__":
    main()
