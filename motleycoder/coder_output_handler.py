from typing import Optional, TYPE_CHECKING

from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.agents import MotleyOutputHandler
from motleycrew.common import logger
from motleycrew.common.exceptions import InvalidOutput

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class CoderOutputHandlerInput(BaseModel):
    question: Optional[str] = Field(
        default=None,
        description="The question to be sent to the user. "
        "Leave empty if you only want to submit your edits.",
    )


class CoderOutputHandler(MotleyOutputHandler):
    _description = (
        "Call this tool when you have finished your edits "
        "OR when you need to ask the user a question. "
        "ONLY COMMUNICATE WITH THE USER THROUGH THIS TOOL. Direct messages will be ignored."
    )
    _args_schema = CoderOutputHandlerInput

    def __init__(self, name: str, coder: "MotleyCrewCoder"):
        self._name = name
        super().__init__()

        self.coder = coder

    def handle_output(self, question: Optional[str] = None):
        question = question or "OK"  # Empty question == OK

        logger.info("Handling output: %s", question)
        reflected_message = self.process_output_and_reflect()

        if reflected_message:
            if self.coder.num_reflections < self.coder.max_reflections:
                self.coder.num_reflections += 1
                return InvalidOutput(reflected_message)
            else:
                logger.warning(f"Only {self.coder.max_reflections} reflections allowed, stopping.")

        if question:
            return question

    def process_output_and_reflect(self):
        edited_files = self.coder.aider_edited_files

        if edited_files and self.coder.auto_test:
            test_errors = self.coder.commands.cmd_test(self.coder.test_cmd)
            self.coder.test_outcome = not test_errors
            if test_errors:
                ok = self.coder.io.confirm_ask("Attempt to fix test errors?")
                if ok:
                    return test_errors
