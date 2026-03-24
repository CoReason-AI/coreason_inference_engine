with open("src/coreason_inference_engine/engine.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "try:" in line and lines[i+1].strip() == 'data = json.loads(payload.decode("utf-8"))':
        # we found the try block. Let's make sure the except block doesn't have an unmatched paren
        for j in range(i, i+15):
            if lines[j].strip() == ")":
                lines[j] = ""
        break

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.writelines(lines)
