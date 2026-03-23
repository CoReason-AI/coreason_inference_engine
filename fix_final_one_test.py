with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("ledger=EpistemicLedgerState(history=[])", "raw_ledger=EpistemicLedgerState(history=[]).model_dump()")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
