def is_user_turn(messages):
    return messages and len(messages) > 0 and messages[-1]["role"] == "user"


def is_assistant_turn(messages):
    return messages and len(messages) > 0 and messages[-1]["role"] == "assistant"


def process_end_eos(msg, eos_token):
    if msg.endswith("\n"):
        msg = msg[:-1]
    if eos_token and msg.endswith(eos_token):
        msg = msg[: -len(eos_token)]
    return msg
