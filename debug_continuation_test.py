import argparse
import json

from llm_base import LLMGenerateConfig
from oai import OpenAiLLM, OpenAiLLMConfig


def run_case(model, max_tokens, temperature, name, messages, base_url, api_key):
    llm = OpenAiLLM(OpenAiLLMConfig(model_name=model, base_url=base_url, api_key=api_key))
    config = LLMGenerateConfig(max_n_tokens=max_tokens, temperature=temperature, logprobs=False, seed=42)
    final_messages = llm.continual_generate(json.loads(json.dumps(messages, ensure_ascii=False)), config)
    raw_text = final_messages[-1]["content"][len(messages[-1]["content"]):]
    result = {
        "case": name,
        "messages": messages,
        "raw_text_repr": repr(raw_text),
        "final_last_message_repr": repr(final_messages[-1]["content"]),
        "usage": {
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3114/v1")
    parser.add_argument("--model", default="/home/lindachuan/huggingface_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18")
    parser.add_argument("--api-key", default="KEY HERE")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--case", choices=["all", "bracket", "topics", "risk"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = {
        "bracket": [
            {"role": "system", "content": "Continue the assistant content naturally."},
            {"role": "assistant", "content": "["},
        ],
        "topics": [
            {"role": "system", "content": "List topics and reasons in the required format only."},
            {"role": "user", "content": "Analyze this claim briefly."},
            {"role": "assistant", "content": "Topics and reasons:\n["},
        ],
        "risk": [
            {"role": "system", "content": "Output only one score in the format Risk Score: [[X]]."},
            {"role": "user", "content": "Original Claim:\n<text>test</text>"},
            {"role": "assistant", "content": "Analysis:\nTopics and reasons:\n[topic]: reason\n\n**RISK LEVEL**: relatively safe"},
            {"role": "assistant", "content": "Risk Score: [["},
        ],
    }
    selected = cases.keys() if args.case == "all" else [args.case]
    for name in selected:
        run_case(args.model, args.max_tokens, args.temperature, name, cases[name], args.base_url, args.api_key)


if __name__ == "__main__":
    main()
