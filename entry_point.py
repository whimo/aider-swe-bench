import logging
import re
import sys
import traceback
from typing import Callable

from langchain_core.prompts import SystemMessagePromptTemplate

from bug_fixer import get_bug_fixer_task
from motleycoder.codemap.file_group import FileGroup
from motleycoder.prompts import MotleyCoderPrompts
from motleycoder.repo import GitRepo
from motleycrew import MotleyCrew
from motleycrew.common import logger, configure_logging


class DualLogger:
    def __init__(self, file_path, logger):
        self.file_path = file_path
        self.logger = logger
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def __enter__(self):
        self._file = open(self.file_path, "a")

        # Redirect stdout and stderr to the custom stream
        sys.stdout = self
        sys.stderr = self

        # Create a file handler that logs to the same file
        self.file_handler = logging.FileHandler(self.file_path)
        self.file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.file_handler.setFormatter(formatter)

        # Add the file handler to the existing logger
        self.logger.addHandler(self.file_handler)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._file.close()

        # Remove the file handler from the logger
        self.logger.removeHandler(self.file_handler)

        # Close the file handler
        self.file_handler.close()

    def write(self, message):
        # Write to the original stdout/stderr and the file, but only non-empty messages
        if message.strip():
            self._original_stdout.write(message)
            self._file.write(message)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()


class PromptsForBenchmark(MotleyCoderPrompts):
    repo_content_prefix = """Here are summaries of some files present in my git repository.
"""

    file_edit_success = """The file {file_path} has been successfully edited.
If you are finished, call the tool `return_to_user` to apply the changes and inform the user that you have finished.
"""

    main_system = SystemMessagePromptTemplate.from_template(
        """Act as an expert software developer.
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.

You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!

Take requests for changes to the supplied code.
If the request is ambiguous, ask questions using the tool `return_to_user`.

Always reply to the user in the same language they are using.

Once you understand the request you MUST:
1. Think step-by-step and explain the needed changes with a numbered list of short sentences.
2. Make the changes to the files by calling the tool `edit_file` with the *SEARCH/REPLACE arguments* for each change. 
You can keep calling the tool with new *SEARCH/REPLACE arguments* until you have made all the necessary changes. 
ONLY EVER RETURN CODE IN THE ARGUMENTS OF THE `edit_file` TOOL CALL!
3. After making all the necessary changes, you MUST call the tool `return_to_user` to apply the changes and to inform 
the user that you have finished. You can't call any tools after this step.

You have access to the following tools:
{tools}

All changes to files must be made using the `edit_file` tool.
"""
    )


def entry_point(
    problem_statement: str,
    repo_path: str,
    existing_test_runner: Callable,
    token_count: Callable,
    llm_name: str | None = None,
    chat_history_file: str | None = None,
):
    configure_logging(verbose=True)

    with DualLogger(chat_history_file, logger):
        try:
            # gold_entity, gold_entity_text = get_gold_entity(repo_map, gold_files, gold_patch)
            #
            # crew = MotleyCrew()
            # file_finder_task = get_file_finder_task(problem_statement, repo_map, crew, llm_name)
            # result = crew.run()
            # output = file_finder_task.output
            # if not isinstance(output, dict) or "entity" not in output or len(output["entity"]) != 2:
            #     print(output)
            #     return None

            # if not output["entity"][1] in set(gold_files):
            #     result_writer(output)
            #     print("ouch!")

            # Now run the bug-fixing task

            repo = GitRepo(repo_path)
            file_group = FileGroup(repo)

            crew = MotleyCrew()
            bug_fixer_task = get_bug_fixer_task(
                repo=repo,
                file_group=file_group,
                problem_statement=problem_statement,
                existing_test_runner=existing_test_runner,
                prompts=PromptsForBenchmark(),
                token_count=token_count,
                crew=crew,
                llm_name=llm_name,
            )
            result = crew.run()
            output2 = bug_fixer_task.output
            if output2 != "Tests passed!":
                return None

            # if gold_entity:
            #     output["gold_entity"] = gold_entity.name
            # output["gold_entity_text"] = gold_entity_text
            # result_writer(output)

            print("yay!")
            return {"files": list(file_group.edited_files), "result": output2}
        except Exception as e:
            logger.error(traceback.format_exc())
            raise e
            return None


# So, the plan is:
# Create a repo map driven by the issue description
# Give the agent a tool to get details for desired objects
# Give the agent an output handler to return a filename and line number?


def extract_line_range(diff_text):
    # Regex pattern to match the line range part of the diff
    line_range_pattern = re.compile(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

    # Find all matches in the diff text
    matches = line_range_pattern.findall(diff_text)

    # Extract the ranges as a list of tuples
    line_ranges = [
        (int(start_old), int(length_old), int(start_new), int(length_new))
        for start_old, length_old, start_new, length_new in matches
    ]

    return line_ranges


def extract_modified_entity(diff_text):
    # Regex pattern to find the modified entity name in the diff
    pattern = re.compile(r"@@.*?@@\s*(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\()")
    match = pattern.search(diff_text)

    if match:
        entity_name = match.group(1).strip()
        return entity_name
    return None


def get_gold_entity(repo_map, gold_files, gold_patch):
    tag_graph = repo_map.get_tag_graph()
    line_ranges = extract_line_range(gold_patch)
    start = min(r[0] for r in line_ranges)
    end = max(r[0] + r[1] - 1 for r in line_ranges)

    best_node = None

    for node in list(tag_graph.nodes):
        if node.rel_fname in gold_files and node.line <= start and node.end_line >= end:
            if best_node is None or (node.end_line - node.line) < (
                best_node.end_line - best_node.line
            ):
                best_node = node

    return best_node, extract_modified_entity(gold_patch)
