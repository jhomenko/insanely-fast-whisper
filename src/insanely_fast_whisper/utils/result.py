from typing import TypedDict


class JsonTranscriptionResult(TypedDict):
    speakers: list
    chunks: list
    text: str


def build_result(transcript, outputs) -> JsonTranscriptionResult:
    if transcript:
        return {
            "speakers": transcript
        }
    else:
        return {
            "chunks": outputs["segments"],
            "text": "".join(segment["text"] for segment in outputs["segments"])
        }
