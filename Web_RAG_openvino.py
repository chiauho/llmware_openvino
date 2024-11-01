# https://github.com/llmware-ai/llmware/blob/main/fast_start/rag/example-3-prompts_and_models.py
# A couple of issues to note:
# Need to pip install openvino openvino-genai
# After this, may still have error with DLL dependencies. So need to resolve this.
# pip install --upgrade openvino-dev
# Microsoft Visual C++ Redistributable is also required

# We will be using llmware quantized models specially tuned for openvino

import time
from llmware.prompts import Prompt
from llmware.models import ModelCatalog
from importlib import util


if not util.find_spec("openvino"):
    print("pip3 install openvino first")
    raise ValueError("openvino not installed. Stopping execution.")
if not util.find_spec("openvino_genai"):
    print("pip3 install openvino_genai first")
    raise ValueError("openvino_genai not installed. Stopping execution.")

print("It's all good")

