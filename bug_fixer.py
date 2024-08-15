from typing import Callable

from langchain_core.tools import render_text_description

from motleycoder.codemap.file_group import FileGroup
from motleycoder.codemap.repomap import RepoMap
from motleycoder.file_edit_tool import FileEditTool
from motleycoder.inspect_entity_tool import InspectEntityTool
from motleycoder.linter import Linter
from motleycoder.prompts import MotleyCoderPrompts
from motleycoder.repo import GitRepo
from motleycoder.user_interatction import UserInterface
from motleycrew import MotleyCrew
from motleycrew.agents import MotleyOutputHandler
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask


def get_bug_fixer_task(
    repo: GitRepo,
    file_group: FileGroup,
    problem_statement: str,
    existing_test_runner: Callable,
    prompts: MotleyCoderPrompts,
    token_count: Callable,
    crew: MotleyCrew,
    llm_name: str | None = None,
) -> SimpleTask:
    if llm_name is None:
        llm = None
    else:
        llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    # mod_fname = entity_to_modify[1]
    # mod_entity = entity_to_modify[0]

    repo_map = RepoMap(
        root=repo.root,
        token_count=token_count,
        repo_content_prefix=prompts.repo_content_prefix,
        file_group=file_group,
        cache_graphs=True,
    )

    repo_map_str = repo_map.repo_map_from_message(problem_statement, llm=llm)

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
The issue was filed some time ago.
The repo has been checked out at the commit that existed at the moment the issue was filed.
If you are already familiar with this repo, be cautious!
You are working with an old version of the repo!
Filenames, directory names, file contents, etc may be different than what you're used to.

Propose changes to update the repo to fix the problem below.
The issue is as follows:

{problem_statement}

Here is a summary of the repo, with a special focus on the files that need to be modified:
{repo_map_str}

You must FIRST identify the entity that needs to be modified and ONLY THEN make changes to the code.

You can use the inspect_entity tool to get more information about specific entities in the repo.
ONLY use the inspect_entity tool as long as NECESSARY to figure out the modifications.
NEVER call the inspect_entity tool more than 5 times.
"""
    user_interface = UserInterface(yes=True)
    linter = Linter()

    inspect_entity_tool = InspectEntityTool(repo_map)
    file_edit_tool = FileEditTool(
        file_group=file_group,
        user_interface=user_interface,
        linter=linter,
        repo_map=repo_map,
        prompts=prompts,
    )

    # TODO: have the output handler write a test for the issue and use it to check the fix?
    tools = [inspect_entity_tool, file_edit_tool]

    class BugFixerOutputHandler(MotleyOutputHandler):
        _name = "return_to_user"
        iteration = 0

        def handle_output(self):
            self.iteration += 1
            if self.iteration >= self.max_iterations:
                return "Tests passed!"

            out = existing_test_runner()
            if out is None:
                return "Tests passed!"
            else:
                raise InvalidOutput("Existing tests failed:\n" + out)

    bug_fixer = ReActToolCallingMotleyAgent(
        name="bug_fixer",
        tools=tools,
        prompt_prefix=prompts.prompt_template.partial(
            tools=render_text_description(tools)
        ),
        output_handler=BugFixerOutputHandler(max_iterations=3),
        chat_history=True,
        verbose=True,
    )

    task = SimpleTask(
        crew=crew,
        name="Apply the fixes to the issue",
        description=message,
        agent=bug_fixer,
    )

    return task
