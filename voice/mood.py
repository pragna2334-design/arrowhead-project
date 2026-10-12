"""Pipecat processor that tags each customer turn with a mood before it reaches the LLM."""

from __future__ import annotations

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from brain.state import CallState
from voice.context import tag_last_user_message


class MoodTagger(FrameProcessor):
    """Sits between the user aggregator and the LLM. Same classifier as text mode."""

    def __init__(self, state: CallState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            # VERIFY (live run): get_messages() returns the stored dicts, so the edit persists.
            mood = tag_last_user_message(frame.context.get_messages())
            if mood:
                self._state.mood = mood
        await self.push_frame(frame, direction)
