import os
from termcolor import colored

from main import PHI2


model_path = "../phi-2"
model_type = "CUDA_FP16"
llm = PHI2(model_path, model_type, max_length=500, temperature=0.6)

terminal_width = os.get_terminal_size().columns


print(colored("Welcome to the Phi-2 LLM demo. Type 'exit' to quit.", "blue"))
print()

while True:
    user_input = input(colored("User:\n", "green"))
    print()

    if user_input == "exit":
        break

    print(colored("LLM:", "magenta"))

    response, response_time = llm.generate(user_input)
    print(f"{response}")

    time_text = f">> Time taken: {response_time:.2f} seconds >>"
    time_text += "-" * (terminal_width - len(time_text))
    print(colored(time_text, "yellow"))
    print()
