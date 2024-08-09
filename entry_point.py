import re
import sys
import traceback
from typing import Callable

from aider.codemap.repomap import RepoMap
from aider.coders.motleycrew_coder.motleycrew_coder import MotleyCrewCoder
from bug_fixer import get_bug_fixer_task
from file_finder import get_file_finder_task
from motleycrew import MotleyCrew
from motleycrew.common import logger, configure_logging


class DualLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def __enter__(self):
        self._file = open(self.file_path, "w")
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._file.close()

    def write(self, message):
        self._original_stdout.write(message)
        self._file.write(message)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()


def entry_point(
    problem_statement: str,
    repo_map: RepoMap,
    coder: MotleyCrewCoder,
    existing_test_runner: Callable,
    result_writer,
    llm_name: str | None = None,
    chat_history_file: str | None = None,
    gold_files: list[str] | None = None,
    gold_patch: str | None = None,
):
    configure_logging(verbose=True)

    with DualLogger(chat_history_file):
        try:
            gold_entity, gold_entity_text = get_gold_entity(repo_map, gold_files, gold_patch)

            crew = MotleyCrew()
            file_finder_task = get_file_finder_task(problem_statement, repo_map, crew, llm_name)
            result = crew.run()
            output = file_finder_task.output
            if not isinstance(output, dict) or "entity" not in output or len(output["entity"]) != 2:
                print(output)
                return None

            # if not output["entity"][1] in set(gold_files):
            #     result_writer(output)
            #     print("ouch!")

            # Now run the bug-fixing task
            crew = MotleyCrew()
            bug_fixer_task = get_bug_fixer_task(
                coder,
                output["entity"],
                problem_statement,
                existing_test_runner,
                repo_map,
                crew,
                llm_name,
            )
            result = crew.run()
            output2 = bug_fixer_task.output
            if output2 != "Tests passed!":
                return None

            if gold_entity:
                output["gold_entity"] = gold_entity.name
            output["gold_entity_text"] = gold_entity_text
            result_writer(output)

            print("yay!")
            return {"files": [output["entity"][1]], "result": output2}
        except Exception as e:
            logger.error(traceback.format_exc())
            # raise e
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
