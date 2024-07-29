import os.path
from typing import List, Dict

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.codemap.repomap import RepoMap, search_terms_from_message
from aider.coders.motleycrew_coder.inspect_object_tool import InspectEntityTool

from motleycrew import MotleyCrew
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common.exceptions import InvalidOutput


def get_file_finder_task(
    problem_statement: str,
    repo_map: RepoMap,
    crew: MotleyCrew,
    llm_name: str | None = None,
) -> SimpleTask:
    if llm_name is None:
        llm = None
    else:
        llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    repo_map_str = repo_map.repo_map_from_message(problem_statement, llm=llm)

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
    The issue was filed some time ago.
    The repo has been checked out at the commit that existed at the moment the issue was filed.
    If you are already familiar with this repo, be cautious!
    You are working with an old version of the repo!
    Filenames, directory names, file contents, etc may be different than what you're used to.
    
    Your task is to identify the Python file that needs to be modified to fix the issue. 
    You are given an initial overview of the repo:
    {repo_map_str}
    
    The issue is as follows:
    {problem_statement}
    
    You can use the inspect_object tool to get more information about specific entities in the repo.
    ONLY use the inspect_object tool as long as NECESSARY to identify the file that needs to be modified.
    NEVER call the inspect_object tool more than 3 times.
    
    Return the filename of the file that needs to be modified, using the output_handler tool. 
    If you can't decide between several possibilities, return all the possible filenames. 
    However, return as few filenames as possible, try to return only one of you can.
    Always return the filename(s) as a LIST of STRINGS, like `["dir1/dir2/file1.py", "file2.py"]`.
    """

    inspect_entity_tool = InspectEntityTool(repo_map)

    def check_files(rel_file_names: List[str]) -> Dict[str, List[str]]:
        for rel_file_name in rel_file_names:
            abs_path = str(repo_map.file_group.abs_root_path(rel_file_name))
            if not rel_file_name.endswith(".py"):
                raise InvalidOutput(f"File {rel_file_name} is not a Python file")
            if not os.path.isfile(abs_path):
                raise InvalidOutput(f"File {abs_path} does not exist or is not a file!")

        return {"files": rel_file_names}

    class FileListToolInput(BaseModel):
        files: list[str] = Field(description="List of relative file paths that should be modified.")

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_files,
        # args_schema=FileListToolInput,
    )

    file_finder = ReActToolCallingMotleyAgent(
        name="File-to-fix_finder",
        tools=[inspect_entity_tool],
        output_handler=output_handler,
        chat_history=True,
        verbose=True,
    )

    task = SimpleTask(
        crew=crew,
        name="Identify the .py file that should be modified to fix the issue",
        description=message,
        agent=file_finder,
    )

    return task
