import os.path
from typing import List, Dict, Any, Callable

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.codemap.repomap import RepoMap, search_terms_from_message
from aider.coders.motleycrew_coder.inspect_entity_tool import InspectEntityTool

from motleycrew import MotleyCrew
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent

from aider.coders.motleycrew_coder.file_edit_tools import get_file_edit_tools
from aider.coders.motleycrew_coder.motleycrew_coder import MotleyCrewCoder


def get_bug_fixer_task(
    coder: MotleyCrewCoder,
    entity_to_modify: List[str],
    problem_statement: str,
    existing_test_runner: Callable,
    repo_map: RepoMap,
    crew: MotleyCrew,
    llm_name: str | None = None,
) -> SimpleTask:
    if llm_name is None:
        llm = None
    else:
        llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    mod_fname = entity_to_modify[1]

    repo_map_str = repo_map.repo_map_from_message(
        problem_statement, rel_added_fnames={mod_fname}, llm=llm
    )

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
    The issue was filed some time ago.
    The repo has been checked out at the commit that existed at the moment the issue was filed.
    If you are already familiar with this repo, be cautious!
    You are working with an old version of the repo!
    Filenames, directory names, file contents, etc may be different than what you're used to.

    The issue is as follows:
    {problem_statement}
    
    You should try to fix the issue by modifying only the following file from the repo:
    {mod_fname}. Most likely, you will need to modify in that file the entity named {entity_to_modify[0]}, 
    but use your judgement to find the best solution. If you're sure you need to modify a different file instead,
    first add that the file containing that entity to the list of files to be modified using the add_files tool.
    
    Here is a summary of the repo, with a special focus on the files that need to be modified:
    {repo_map_str}
    
    You can use the inspect_entity tool to get more information about specific entities in the repo.
    ONLY use the inspect_entity tool as long as NECESSARY to identify the file that needs to be modified.
    NEVER call the inspect_entity tool more than 3 times.
    
    Your task is to fix the issue by modifying one of the files listed above, using the edit_file tool.
    """

    inspect_entity_tool = InspectEntityTool(repo_map)

    add_files_tool, get_modifiable_files_tool, file_edit_tool = get_file_edit_tools(coder)

    # TODO: have the output_handler run existing tests!
    # TODO: have the output handler write a test for the issue and use it to check the fix?
    add_files_tool.add_files([mod_fname])
    tools = [inspect_entity_tool, file_edit_tool, add_files_tool, get_modifiable_files_tool]

    def check_tests():
        return "Tests passed!"
        out = existing_test_runner()
        if out is None:
            return "Tests passed!"
        else:
            raise InvalidOutput("Existing tests failed:\n" + out)

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_tests,
    )

    file_finder = ReActToolCallingMotleyAgent(
        name="File-to-fix_finder",
        tools=tools,
        # TODO: is this the right way to inject the fake chat history?
        prompt_prefix=coder.create_prompt("").partial(
            tools=tools
        ),  # get usage examples as fake chat history
        output_handler=output_handler,
        chat_history=True,
        verbose=True,
    )

    task = SimpleTask(
        crew=crew,
        name="Apply the fixes to the issue",
        description=message,
        agent=file_finder,
    )

    return task
