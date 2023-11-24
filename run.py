
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        offload_folder="offload",
    )

    model.seqlen = model.config.max_position_embeddings 
    return model


def prune(
    model: str,
    seed: int,
    nsamples: int, 
    sparsity_ratio: float, 
    sparsity_type,
    prune_method: str,
    cache_dir: str,
    use_variant,
    save: str,
    save_model: str,
    eval_zero_shot
    ):
    """
    model: str, model_id
    seed: int, default =0
    nsamples: int, default=128
    sparsity_ratio: float
    sparsity_type: str choices=["unstructured", "4:8", "2:4"]
    prune_method: str choices=["magnitude", "wanda", "sparsegpt", "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"]
    cache_dir: str
    save: str, ath to save results.
    save_model: str, Path to save the pruned model.


  """
    # Setting seeds for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        assert sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, sparsity_type.split(":"))

    model_name = model.split("/")[-1]
    print(f"loading llm model {model}")
    model = get_llm(model, cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in model or "65b" in model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if sparsity_ratio != 0:
        print("pruning starts")
        if prune_method == "wanda":
            prune_wanda(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "magnitude":
            prune_magnitude(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif prune_method == "sparsegpt":
            prune_sparsegpt(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in prune_method:
            prune_ablate(model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)

    ppl_test = eval_ppl(model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(save):
        os.makedirs(save)
    save_filepath = os.path.join(save, f"log_{prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if eval_zero_shot:
        accelerate=False
        if "30b" in model or "65b" in model or "70b" in model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if save_model:
        model.save_pretrained(save_model, safe_serialization=True)
        tokenizer.save_pretrained(save_model)
