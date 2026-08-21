from src.models import Constrained
from src.loading import load_prompts, parse_json
# from llm_sdk import Small_LLM_Model


def main() -> None:
    # for moulinette
    # try:
    #     prompts = load_prompts()
    #     for prompt in prompts:
    #         decoder = Constrained(prompt=prompt.prompt)
    #         name_token = decoder.get_name()
    #         llm_response = decoder.get_params(name_token)
    #         parse_json(llm_response)
    # except Exception as e:
    #     print(e)
    try:
        prompt = load_prompts()[0].prompt
        decoder = Constrained(prompt=prompt)
        name_token = decoder.get_name()
        llm_response = decoder.get_params(name_token)
        parse_json(llm_response)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
