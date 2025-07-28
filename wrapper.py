import os
import subprocess


def transcribe_with_nvidia_asr(audio_path: str, api_key: str) -> str:
    command = [
        "python",
        "asr_client.py",  # Replace this with the actual path if needed
        "--server",
        "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata",
        "function-id",
        "ee8dc628-76de-4acc-8595-1836e7e857bd",
        "--metadata",
        "authorization",
        f"Bearer {api_key}",
        "--language-code",
        "en-US",
        "--input-file",
        audio_path,
    ]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("ASR error:", result.stderr)
        raise RuntimeError("ASR transcription failed.")

    return result.stdout.strip()
