from llm_sdk import Small_LLM_Model
from .JsonValidator import TestCaseSchema, FunctionDefSchema
from typing import Dict, List, Any
from json.decoder import JSONDecodeError
from pydantic import ValidationError
import json
from sys import exit, stderr

def test():
    prompts = []
    with open("data/input/function_calling_tests.json", 'r') as f:
        content = json.load(f)
        for prompt in content:
            valid = TestCaseSchema(**prompt)
            prompts.append(valid.prompt)

    func_definition: List[FunctionDefSchema] = []
    with open("data/input/functions_definition.json", 'r') as f:
        content = json.load(f)
        for funcdef in content:
            valid = FunctionDefSchema(**funcdef)
            func_definition.append(valid)

    # tes = Small_LLM_Model()
    # string = "what does the sum of 4 and 5"
    # encoded = tes.encode(string).tolist()
    # encoded = encoded[0]
    # with open('tokens.txt', 'w') as f:
    #     print(encoded, file=f)
    #     print("\n", file=f)
    # while True:
    #     tokens = tes.get_logits_from_input_ids(encoded)
    #     new_token = tokens.index(max(tokens))
    #     encoded.append(new_token)
    #     print(tes.decode([new_token]), end="")
def main() -> None:
    try:
        test()
    except JSONDecodeError as e:
        print(f"Invalid Formate For JSON File: {e}", file=stderr)
        exit(1)
    except ValidationError as e:
        print(f"Invalid Data: {e.errors()[0]["msg"]}", file=stderr)
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except (BaseException, Exception) as e:
        print(f"Program Failed [{e}]", file=stderr)
        exit(1)
    exit(0)
