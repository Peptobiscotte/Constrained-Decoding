from pydantic import BaseModel, PrivateAttr, field_validator
from enum import Enum
from llm_sdk import Small_LLM_Model
from typing import TypedDict, Any
import json


class EnumTypes(Enum):
    """Enum class for various types checking."""

    STR = "string"
    NBR = "number"
    BOOL = "boolean"


class TypeDef(BaseModel):
    """Function type class for pydantic checking."""

    type: EnumTypes


class FunctionDef(BaseModel):
    """Function definition class for pydantic checking."""

    name: str
    description: str
    parameters: dict[str, TypeDef]
    returns: TypeDef


class Prompt(BaseModel):
    """Prompt class for pydantic checking."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_pr(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value


class FunctionCallDict(TypedDict):
    """Output class for pydantic checking."""

    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


class Step(BaseModel):
    """Defines content of a step in decoding."""

    fixed: bool
    content: list[int]


class Constrained(BaseModel):
    """Main project class, gets the prompt as context then can use methods
    to get the name of the selected function and its parameters
    """

    prompt: str
    current_number: str = ""
    first_s_turn: bool = True
    regex: bool = False
    _model: Small_LLM_Model = PrivateAttr()
    _func_list: list[FunctionDef] = PrivateAttr()
    _vocab: dict[Any, Any] = PrivateAttr()

    def model_post_init(self, _context: object) -> None:
        from src.loading import load_def, load_vocab
        self._model = Small_LLM_Model()
        self._func_list = load_def()
        self._vocab = load_vocab(self._model.get_path_to_vocab_file())
        self.current_number = ""
        self.first_s_turn = True
        self.regex = False

    def is_valid_n_prefix(self, candidate_s: str) -> bool:
        if candidate_s == "":
            return False
        if candidate_s[0] == "-":
            if len(candidate_s) == 1:
                return True
            if len(candidate_s) > 2 and candidate_s[1] == "0":
                if candidate_s[2].isdigit():
                    return False
            if candidate_s[1:].isdigit():
                return True
            if "." in candidate_s:
                before, after = candidate_s[1:].split(".", 1)
                if before.isdigit() and not after:
                    return True
                if before.isdigit() and after.isdigit():
                    return True
            return False
        if len(candidate_s) > 1 and candidate_s[0] == "0":
            if candidate_s[1].isdigit():
                return False
        if candidate_s.count(".") > 1:
            return False
        if candidate_s.startswith("."):
            return False
        if "." in candidate_s:
            before, after = candidate_s.split(".", 1)
            if before.isdigit() and not after:
                return True
            if before.isdigit() and after.isdigit():
                return True
        if candidate_s.isdigit():
            return True
        return False

    def is_valid_num(self, candidate_s: str) -> bool:
        if not candidate_s:
            return False
        if candidate_s.isdigit():
            return True
        if candidate_s[0] == "-":
            if "." in candidate_s:
                before, after = candidate_s[1:].split(".", 1)
                if before.isdigit() and after.isdigit():
                    return True
        if "." in candidate_s:
            before, after = candidate_s.split(".", 1)
            if before.isdigit() and after.isdigit():
                return True
        return False

    def is_valid_s_prefix(self, candidate_s: str) -> bool:
        if self.first_s_turn and candidate_s != "\"":
            return False
        if not candidate_s.startswith("\""):
            return False
        if candidate_s.count("\"") == 2:
            if not candidate_s.endswith("\""):
                return False
        if candidate_s.count("\"") > 2:
            return False
        # if self.regex:
        #     if not self.is_valid_regex(candidate_s):
        #         return False
        return True

    def is_valid_str(self, candidate_s: str) -> bool:
        if not candidate_s.startswith("\""):
            return False
        if not candidate_s.endswith("\""):
            return False
        if not candidate_s.count("\"") == 2:
            return False
        return True

    def is_valid_regex(self, candidate_s: str, curr: str) -> bool:
        # print(f"REGEX: {curr}, CANDIDATE: {candidate_s}")
        if "\\" in candidate_s:
            return False
        if len(curr) > 10:
            return False
        return True

    def terminate_num(self, candidate_s: str) -> bool:
        if "." in candidate_s:
            return True
        return False

    def get_name(self) -> list[int]:
        func_list = self._func_list
        text = ""
        model = self._model
        vocab = self._vocab
        func_name_tok: list[list[int]] = []
        func_name_list: list[tuple[str, str]] = []
        # encoding name into token list and updating prompt
        for func in func_list:
            tokens: list[int] = []
            func_name_list.append((func.name, func.description))
            for token in model.encode(func.name)[0]:
                tokens.append(int(token.item()))
            func_name_tok.append(tokens)
            text += func.name + "\n" + func.description + "\n\n"

        # updating the prompt w additional values
        text += ("Available functions:\n")
        for name, desc in func_name_list:
            text += (f"{name}: {desc}\n\n")
        text += (
            "user request: " + json.dumps(self.prompt) + "\n\n"
            + "Replace means Substitute\n"
            + "Which function handles the user request?" + "\n"
            + "Return its name"
            )

        ids = model.encode(text)  # text to token Tensor

        token_list: list[int] = []

        for token_id in ids[0]:  # transform Tensor into list
            token_list.append(int(token_id.item()))

        response: list[int] = []
        # llm constrained decoding loop
        while True:
            # get priolist from token_list (our tokenized prompt)
            logits = model.get_logits_from_input_ids(token_list)

            for id, _ in enumerate(logits):
                if id not in vocab:
                    logits[id] = float("-inf")
                    continue
                token = vocab[id]
                # get rid of special invisible control tokens
                if token == "":
                    logits[id] = float("-inf")
                    continue
                # add previous res + current token
                candidate = response.copy()
                candidate.append(id)
                # if no crossing set to -inf
                c_len = len(candidate)
                if not any(
                        t_list[:c_len] == candidate for
                        t_list in func_name_tok):
                    logits[id] = float("-inf")
            # # get the best priority token in our list
            best_id, best_logit = max(enumerate(logits), key=lambda x: x[1])
            # # break if empty
            if best_logit == float("-inf"):
                break
            # # append the token to build our response
            response.append(best_id)
            # # append to our token_list from which llm creates logits
            token_list.append(best_id)
            # # if response corresponds to full name in list, end loop
            if any(name == response for name in func_name_tok):
                break

        return response

    def get_params(self, name: list[int]) -> str:
        selected_fn_name = self._model.decode(name)
        # params_list for prompt
        params_list: list[tuple[str, str]] = []
        terminator_list: list[int] = []
        for token in self._model.encode(".0")[0]:
            terminator_list.append(int(token.item()))
        # get first elem match in iterator
        selected_fn = next(
            func for func in self._func_list if func.name == selected_fn_name
            )
        # building our schema based on selected function
        steps: list[Step] = []
        start = ('{"prompt": '
                 + json.dumps(self.prompt)
                 + ', "name": "'
                 + selected_fn_name
                 + '", "parameters": {'
                 )
        # fill the first fixed Step
        tokens: list[int] = []
        for token in self._model.encode(start)[0]:
            tokens.append(int(token.item()))
        steps.append(Step(fixed=True, content=tokens))

        nb_params = len(selected_fn.parameters)
        # fill steps with fixed content and not fixed containing
        # the type we expect from the LLM
        for params, types in selected_fn.parameters.items():
            params_list.append((params, types.type.value))
            s = '"' + params + '": '
            tokens = []
            for token in self._model.encode(s)[0]:
                tokens.append(int(token.item()))
            steps.append(Step(fixed=True, content=tokens))
            tokens = []
            for token in self._model.encode(types.type.value)[0]:
                tokens.append(int(token.item()))
            steps.append(Step(fixed=False, content=tokens))
            nb_params -= 1
            if nb_params == 0:
                tokens = []
                for token in self._model.encode("}}")[0]:
                    tokens.append(int(token.item()))
                steps.append(Step(fixed=True, content=tokens))
            else:
                tokens = []
                for token in self._model.encode(", ")[0]:
                    tokens.append(int(token.item()))
                steps.append(Step(fixed=True, content=tokens))

        # create the prompt
        text = (
                "<|im_start|>system\n"
                "Extract the arguments for the selected function from the "
                "request. "
                "Use values explicitly present in the request. "
                "Do not rename, describe, or invent values. /no_think "
                "For regex: keep simple, no control characters,\n"
                "Create the shortest regex possible"
                "<|im_end|>\n"

                "<|im_start|>user\n"
                f"request: {self.prompt}\n\n"
                "Selected function:\n"
                f"{selected_fn_name}\n"
                f"Description: {selected_fn.description}.\n"
                "Required arguments:\n"
        )
        for param, typess in params_list:
            text += (
                f"- {param}: {typess}\n"
                f'Extract the value of {param} from the user request.\n\n')
        text += (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            )
        encoded: list[int] = []
        for token in self._model.encode(text)[0]:
            encoded.append(int(token.item()))

        response: list[int] = []
        current_response: list[int] = []

        current_step = 0
        # creates list for true and false
        true_list: list[int] = []
        for token in self._model.encode("true")[0]:
            true_list.append(int(token.item()))
        false_list: list[int] = []
        for token in self._model.encode("false")[0]:
            false_list.append(int(token.item()))

        while current_step < len(steps):
            fixed = steps[current_step].fixed
            content = steps[current_step].content
            logits = self._model.get_logits_from_input_ids(encoded)
            v_type = self._model.decode(content)
            # HELPFUL PRINT FOR DEBUG
            # print("CONTENT: ", self.model.decode(content))
            print(self.regex)
            for id, _ in enumerate(logits):
                if id not in self._vocab:
                    logits[id] = float("-inf")
                    continue
                token = self._vocab[id]
                # get rid of special invisible control tokens
                if token == "":
                    logits[id] = float("-inf")
                    continue
                candidate = current_response.copy()
                if self.regex:
                    candidate_s = self._model.decode([id])
                    curr = self._model.decode(candidate)
                    if (
                        not self.is_valid_regex(candidate_s, curr) and
                        id != 1
                    ):
                        logits[id] = float("-inf")
                candidate.append(id)
                c_len = len(candidate)
                # if no crossing set to -inf
                if fixed:
                    if not content[:c_len] == candidate:
                        logits[id] = float("-inf")
                else:
                    if v_type == "string":
                        if id == 1:
                            print("BEFORE", logits[id])
                        candidate_s = self._model.decode(candidate)
                        if not self.is_valid_s_prefix(candidate_s) and (
                            not self.is_valid_str(candidate_s)
                                or steps[
                                    current_step + 1
                                    ].content[0] != candidate[-1]
                                ):
                            logits[id] = float("-inf")
                    # for numbers, we check two conditions, is the candidate
                    # a good prefix for a number (1. -> True) and
                    # is the current number valid (1. -> False) or is the
                    # candidate start of the next step.
                    elif v_type == "number":
                        candidate_s = self._model.decode(candidate)
                        if not self.is_valid_n_prefix(candidate_s) and (
                            not self.is_valid_num(self.current_number)
                                or steps[
                                    current_step + 1
                                    ].content[0] != candidate[-1]
                                ):
                            logits[id] = float("-inf")
                    # for bools we simply verify that the candidate is
                    # a prefix of tokenized True or False
                    elif v_type == "boolean":
                        if (
                            true_list[:c_len] != candidate and
                            false_list[:c_len] != candidate
                        ):
                            logits[id] = float("-inf")
            # # get the best priority token in our list
            best_id, best_logit = max(
                enumerate(logits), key=lambda x: x[1])
            if v_type == "string":
                if self.first_s_turn:
                    self.first_s_turn = False
                top_20 = sorted(
                    enumerate(logits),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                for tok, logit in top_20:
                    print(self._model.decode([tok]), " ", logit)
            # # break if empty
            if best_logit == float("-inf"):
                break
            # for a number we always get the first token of the next step
            # in our max concurrency, if it wins, it means the LLM
            # chose it as a next token so the number generation is
            # over, we continue the next step.
            if v_type == "number" and (
                    best_id == steps[current_step + 1].content[0]):
                if not self.terminate_num(self.current_number):
                    current_response.append(13)
                    current_response.append(15)
                    response.append(13)
                    response.append(15)
                    encoded.append(13)
                    encoded.append(15)
                current_step += 1
                current_response = []
                continue
            # append the token to build our response
            current_response.append(best_id)
            response.append(best_id)
            encoded.append(best_id)
            # for a string we force the llm to chose an ending string that
            # doesn't cross the next step so if we got an entire string
            # we can move to next step
            if v_type == "string" and self.is_valid_str(
                    self._model.decode(current_response)):
                current_step += 1
                current_response = []
                self.first_s_turn = True
                self.regex = False
            # append to our encoded from which llm creates logits
            if v_type == "number":
                self.current_number = self._model.decode(current_response)
            # HELPFUL PRINT FOR DEBUG
            print("RESPONSE: ", self._model.decode(response))
            # if response corresponds to full content in list, end loop
            if fixed and content == current_response:
                if v_type == "\"regex\": ":
                    self.regex = True
                current_step += 1
                current_response = []
            # if bool and corresponds to True or False as tokens list
            # go to next step
            if not fixed and v_type == "boolean" and (
                current_response == true_list or
                current_response == false_list
            ):
                current_step += 1
                current_response = []

        # print(steps[0].content)
        # print(encoded)
        # print(self.model.decode(encoded))
        # print(self.model.decode(response))
        return self._model.decode(response)
