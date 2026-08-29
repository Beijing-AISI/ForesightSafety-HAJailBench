import argparse
import concurrent.futures
import json
import queue
import threading
from copy import deepcopy
from pathlib import Path

from llm_base import LLMGenerateConfig
from multi_agent_judge_core import MultiAgentJudge, MultiAgentJudgeConfig
from oai import OpenAiLLMConfig


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_output_complete(data, existing):
    if isinstance(data, list) and isinstance(existing, list):
        return len(existing) == len(data) and all(isinstance(item, dict) and "score" in item for item in existing)
    if isinstance(data, dict) and "results" in data and isinstance(existing, dict) and "results" in existing:
        results = data["results"]
        existing_results = existing["results"]
        return len(existing_results) == len(results) and all(isinstance(item, dict) and "score" in item for item in existing_results)
    return False


def preprocess_messages(messages):
    output = []
    for message in deepcopy(messages):
        content = message.get("content")
        if content is None:
            content = ""
        else:
            content = str(content)[:1200]
        output.append({"role": message["role"], "content": content})
    return output


def extract_goal(item):
    return "" if item.get("goal") is None else str(item.get("goal"))


def extract_messages(item):
    if "messages" in item:
        return preprocess_messages(item["messages"])
    if "data" in item and item["data"]:
        return preprocess_messages(item["data"][0]["messages"])
    raise ValueError(f"Unsupported item structure: {item.keys()}")


def judge_item(judge, item):
    item = json.loads(json.dumps(item, ensure_ascii=False))
    score_compound = judge.judge(extract_messages(item), extract_goal(item))
    item["score"] = score_compound["score"]
    item["aspects"] = score_compound.get("points")
    item["prompt_tokens"] = score_compound.get("prompt_tokens", 0)
    item["completion_tokens"] = score_compound.get("completion_tokens", 0)
    return item


class ProgressLog:
    def __init__(self, path, workers):
        self.path = path
        self.lines = [""] * workers
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()

    def set(self, slot, text):
        with self.lock:
            self.lines[slot] = text
            self.flush()

    def clear(self, slot):
        self.set(slot, "")

    def flush(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
            f.write("\n")


def format_progress_line(rel_path, file_index, file_total, item_index, item_total, percent):
    label = str(rel_path).replace("\\", "/")
    file_part = f"f {file_index}/{file_total}" if file_total else "f 0/0"
    item_part = f"i {item_index}/{item_total}" if item_total else "i 0/0"
    return f"{label.ljust(55)}{file_part} {item_part} {percent}%"


def make_thread_local_runner(judge_factory):
    local_state = threading.local()

    def run(item):
        if not hasattr(local_state, "judge"):
            local_state.judge = judge_factory()
        return judge_item(local_state.judge, item)

    return run


def process_items(items, judge_factory, item_workers, progress_log, slot, rel_path, file_index, file_total):
    total = len(items)
    if total == 0:
        progress_log.set(slot, format_progress_line(rel_path, file_index, file_total, 0, 0, 100))
        return []
    progress_log.set(slot, format_progress_line(rel_path, file_index, file_total, 0, total, 0))
    if item_workers <= 1:
        judge = judge_factory()
        output = []
        for index, item in enumerate(items, 1):
            output.append(judge_item(judge, item))
            progress_log.set(slot, format_progress_line(rel_path, file_index, file_total, index, total, round(index * 100 / total)))
        return output
    output = [None] * total
    completed = 0
    runner = make_thread_local_runner(judge_factory)
    with concurrent.futures.ThreadPoolExecutor(max_workers=item_workers) as executor:
        futures = {executor.submit(runner, item): index for index, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            output[index] = future.result()
            completed += 1
            progress_log.set(slot, format_progress_line(rel_path, file_index, file_total, completed, total, round(completed * 100 / total)))
    return output


def process_file_with_progress(judge_factory, input_path, output_path, item_workers, progress_log, slot, rel_path, file_index, file_total):
    data = read_json(input_path)
    if output_path.exists():
        try:
            existing = read_json(output_path)
            if is_output_complete(data, existing):
                print(f"skip {output_path}")
                return
        except Exception:
            pass
    if isinstance(data, list):
        output = process_items(data, judge_factory, item_workers, progress_log, slot, rel_path, file_index, file_total)
    elif isinstance(data, dict) and "results" in data:
        output = data
        output["results"] = process_items(data["results"], judge_factory, item_workers, progress_log, slot, rel_path, file_index, file_total)
    else:
        raise ValueError(str(input_path))
    write_json(output_path, output)
    progress_log.clear(slot)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="HAJailbench")
    parser.add_argument("--output-root", default="output/HAJailbench")
    parser.add_argument("--base-url", default="http://127.0.0.1:3114/v1")
    parser.add_argument("--model", default="/home/lindachuan/huggingface_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--item-workers", type=int, default=3)
    parser.add_argument("--debate-round", type=int, default=3)
    parser.add_argument("--disable-refuse", action="store_true")
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--predefined-alignment", action="store_true")
    parser.add_argument("--hardcoded-alignment", action="store_true")
    parser.add_argument("--disable-denoise", action="store_true")
    return parser.parse_args()


def build_judge_factory(args):
    def judge_factory():
        llm_config = OpenAiLLMConfig(llm_type="OpenAiLLM", model_name=args.model, base_url=args.base_url)
        llm_gen_config = LLMGenerateConfig(max_n_tokens=1000, temperature=0.0, logprobs=False, seed=42, stream=False)
        config = MultiAgentJudgeConfig(
            judge_llm_config=llm_config,
            judge_llm_gen_config=llm_gen_config,
            enable_refuse=not args.disable_refuse,
            enable_alignment=not args.disable_alignment,
            enable_predefined_alignment=args.predefined_alignment,
            enable_hardcoded_alignment=args.hardcoded_alignment,
            enable_denoise_filter=not args.disable_denoise,
            debate_round=args.debate_round,
        )
        return MultiAgentJudge(config)

    return judge_factory


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = root / input_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root

    judge_factory = build_judge_factory(args)
    tasks = []
    for input_path in sorted(input_root.rglob("*.json")):
        output_path = output_root / input_path.relative_to(input_root)
        if output_path.exists():
            try:
                if is_output_complete(read_json(input_path), read_json(output_path)):
                    print(f"skip {output_path}")
                    continue
            except Exception:
                pass
        tasks.append((input_path, output_path))
    indexed_tasks = [(index, len(tasks), input_path, output_path) for index, (input_path, output_path) in enumerate(tasks, 1)]

    if args.workers <= 1:
        progress_log = ProgressLog(output_root / "_progress.log", 1)
        for file_index, file_total, input_path, output_path in indexed_tasks:
            process_file_with_progress(judge_factory, input_path, output_path, args.item_workers, progress_log, 0, input_path.relative_to(input_root), file_index, file_total)
        return

    progress_log = ProgressLog(output_root / "_progress.log", args.workers)
    task_queue = queue.Queue()
    for task in indexed_tasks:
        task_queue.put(task)
    errors = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()

    def worker(slot):
        while not stop_event.is_set():
            try:
                file_index, file_total, input_path, output_path = task_queue.get_nowait()
            except queue.Empty:
                progress_log.clear(slot)
                return
            try:
                process_file_with_progress(judge_factory, input_path, output_path, args.item_workers, progress_log, slot, input_path.relative_to(input_root), file_index, file_total)
            except Exception as exc:
                with errors_lock:
                    errors.append((str(input_path), repr(exc)))
                stop_event.set()
                progress_log.clear(slot)
                return

    threads = [threading.Thread(target=worker, args=(slot,), daemon=True) for slot in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(str(errors[0]))


if __name__ == "__main__":
    main()
