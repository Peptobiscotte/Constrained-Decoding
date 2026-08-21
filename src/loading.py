import json
from src.models import FunctionDef, Prompt, FunctionCallDict
from typing import Any
from pathlib import Path


def load_def() -> list[FunctionDef]:
    with open("data/input/functions_definition.json", "r") as file:
        data = json.load(file)

    func_list: list[FunctionDef] = []
    for obj in data:
        func_list.append(FunctionDef(**obj))

    return func_list


def load_prompts() -> list[Prompt]:
    with open("data/input/function_calling_tests.json", "r") as file:
        data = json.load(file)
    prompt_list: list[Prompt] = []
    for obj in data:
        prompt_list.append(Prompt(**obj))

    return prompt_list


def load_vocab(path: str) -> dict[Any, Any]:
    with open(path, "r") as file:
        data = json.load(file)
    id_to_token = {token_id: token for token, token_id in data.items()}
    return id_to_token


def parse_json(resp: str) -> None:
    resp_json: FunctionCallDict = json.loads(resp)
    path = Path("data/output/function_calling_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data: list[Any] = []

    if path.exists():
        with open(path, "r") as file:
            data = json.load(file)

    data.append(resp_json)

    with open(path, "w") as file:
        json.dump(data, file, indent=4)
