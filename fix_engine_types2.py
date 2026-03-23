with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    """    def _extract_latent_traces(
        self, raw_output: str, node: LocalAgentNodeProfile
    ) -> tuple[str, LatentScratchpadReceipt | None]:""",
    """    def _extract_latent_traces(
        self, raw_output: str, node: LocalAgentNodeProfile
    ) -> tuple[str, dict[str, Any] | None]:""",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
