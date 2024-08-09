import os.path
from typing import Dict

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.codemap.repomap import RepoMap
from aider.coders.motleycrew_coder.inspect_entity_tool import InspectEntityTool
from motleycrew import MotleyCrew
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask


class FileListToolInput(BaseModel):
    entity_name: str = Field(description="Name of the entity to modify.", default=None)
    file_name: str = Field(description="Full name of the file containing the entity.", default=None)


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
    
    Your task is to identify the Python file, and the entity name 
    within that file, that needs to be modified to fix the issue. 
    You are given an initial overview of the repo:
    {repo_map_str}
    
    The issue is as follows:
    {problem_statement}
    
    You can use the inspect_entity tool to get more information about specific entities in the repo.
    ONLY use the inspect_entity tool as long as NECESSARY to identify the file that needs to be modified.
    NEVER call the inspect_entity tool more than 5 times.
    NEVER make the same call to the inspect_entity tool more than once.

    Make sure to inspect the entity you're returning using the inspect_entity tool before returning it.
    
    Return the entity name and the filename in which it is defined, that needs to be modified to fix the issue, 
    using the output_handler tool. This should be done in the same format as for the inspect_entity tool, except
    you MUST specify both the entity name and the filename.
    """

    inspect_entity_tool = InspectEntityTool(repo_map)

    def check_entity(entity_name: str, file_name: str) -> Dict:
        abs_path = str(repo_map.file_group.abs_root_path(file_name))
        if not file_name.endswith(".py"):
            raise InvalidOutput(f"File {file_name} is not a Python file")
        if not os.path.isfile(abs_path):
            raise InvalidOutput(f"File {abs_path} does not exist or is not a file!")

        # TODO: check that entity exists in file
        return {"entity": (entity_name, file_name)}

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_entity,
        args_schema=FileListToolInput,
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
