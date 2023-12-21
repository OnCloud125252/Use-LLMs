import os
import time
import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

from _types import Models


terminal_width = os.get_terminal_size().columns


class PHI2:
    def __init__(
        self, model_path: str, model_type: Models, max_length=1000, temperature=0.8
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.max_length = max_length
        self.temperature = temperature

        if model_type not in [
            "CUDA_FP16_Flash-Attention",
            "CUDA_FP16",
            "CUDA_FP32",
            "CPU_FP32",
        ]:
            raise ValueError(
                "Invalid model type. Supported types are: CUDA_FP16_Flash-Attention, CUDA_FP16, CUDA_FP32, CPU_FP32"
            )

        print(colored(f"Loading model from '{model_path}'", "blue"))
        match model_type:
            case "CUDA_FP16_Flash-Attention":
                print(colored("Using CUDA FP16 Flash-Attention", "blue"))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    flash_attn=True,
                    flash_rotary=True,
                    fused_dense=True,
                    device_map="cuda",
                    trust_remote_code=True,
                )
            case "CUDA_FP16":
                print(colored("Using CUDA FP16", "blue"))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="cuda",
                    trust_remote_code=True,
                )
            case "CUDA_FP32":
                print(colored("Using CUDA FP32", "blue"))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cuda",
                    trust_remote_code=True,
                )
            case "CPU_FP32":
                print(colored("Using CPU FP32", "blue"))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )
        print(colored("Model loaded", "green"))
        print("=" * terminal_width)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2", trust_remote_code=True
        )

    @staticmethod
    def purge_llm_output(user_input, llm_output):
        llm_output = llm_output[len(user_input) :]
        parts = llm_output.split("\n<|endoftext|>")
        if len(parts) > 1:
            return parts[0]
        else:
            return llm_output

    def generate(self, user_input):
        time_start = time.time()

        if self.model_type == "CPU_FP32":
            torch.set_default_device("cpu")
        else:
            torch.set_default_device("cuda")

        # TODO: Prompting
        # user_input = f"When appropriate, please utilize Markdown features such as headers, lists, emphasis (e.g., bold and italic), links, images, code blocks in all the outputs.\n\n"
        user_input += f"Instruct:\n{user_input}\nOutput:\n"

        # Tokenize the user input
        llm_input = self.tokenizer(
            user_input, return_tensors="pt", return_attention_mask=False
        )

        # Generate a response from the model
        llm_outputs = self.model.generate(
            **llm_input,
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
        )

        # Decode and purge the response
        llm_response = self.tokenizer.batch_decode(llm_outputs)[0]
        llm_response = self.purge_llm_output(user_input, llm_response)

        time_end = time.time()

        return llm_response, time_end - time_start
