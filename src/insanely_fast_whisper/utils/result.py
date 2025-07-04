from typing import TypedDict


class JsonTranscriptionResult(TypedDict):
    speakers: list
    chunks: list
    text: str


def build_result(transcript, outputs) -> JsonTranscriptionResult:
    if transcript:
        return {
            "speakers": transcript,
            "chunks": transcript,  # Use the diarized transcript as chunks
            "text": outputs.get("text", "")
        }
    else:
        return {
            "chunks": outputs.get("chunks", []),
            "text": outputs.get("text", "")
        }
