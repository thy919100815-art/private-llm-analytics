import json

transcript_path = r"C:\Users\user\.gemini\antigravity\brain\34755a55-2b79-4158-81d9-e0a152897d72\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    last_user_input = None
    for line in f:
        if '"USER_INPUT"' in line:
            obj = json.loads(line)
            if obj.get("type") == "USER_INPUT":
                last_user_input = obj

if last_user_input:
    print(f"Found step {last_user_input.get('step_index')}")
    with open("scratch/latest_prompt.txt", "w", encoding="utf-8") as out:
        out.write(last_user_input.get("content", ""))
    print("Saved to scratch/latest_prompt.txt")
else:
    print("Not found")
