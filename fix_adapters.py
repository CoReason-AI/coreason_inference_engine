import glob

for file in glob.glob("src/coreason_inference_engine/adapters/*.py"):
    with open(file) as f:
        content = f.read()

    # Apply PeftAdapterContract -> dict[str, Any]
    content = content.replace("list[PeftAdapterContract]", "list[dict[str, Any]]")

    # ComputeRateContract -> dict[str, Any]
    content = content.replace("ComputeRateContract | None", "dict[str, Any] | None")

    # generate_stream LatentScratchpadReceipt -> dict[str, Any]
    content = content.replace("LatentScratchpadReceipt | None", "dict[str, Any] | None")

    with open(file, "w") as f:
        f.write(content)
