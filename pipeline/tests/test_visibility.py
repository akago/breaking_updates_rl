import torch

print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count() >= 1:
    torch.cuda.set_device(0)
    print("current_device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

from pipeline.llms.Gemma3 import Gemma3


llm = Gemma3()

llm.generate