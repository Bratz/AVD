with open("banking_judge_llm/core/llm/judge.py", "rb") as f:
    content = f.read()
    if b"\x00" in content:
        print("Null bytes found in file")