import colorsys
import os
import random
import warnings
from typing import List, Set

from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm

from aider.codemap.file_group import (
    FileGroup,
    get_ident_mentions,
    get_ident_filename_matches,
)
from aider.codemap.graph import TagGraph, build_tag_graph, only_defs  # noqa: F402
from aider.codemap.parse import get_tags_raw, read_text  # noqa: F402
from aider.codemap.rank import rank_tags_new, rank_tags  # noqa: F402
from aider.codemap.render import RenderCode

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    cache_missing = False

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        token_count=None,
        repo_content_prefix=None,
        verbose=False,
        file_group: FileGroup = None,
        use_old_ranking: bool = False,
        cache_graphs: bool = False,
    ):
        self.verbose = verbose
        self.use_old_ranking = use_old_ranking

        if not root:
            root = os.getcwd()
        self.root = root

        # self.load_tags_cache()

        self.max_map_tokens = map_tokens

        self.token_count = token_count
        self.repo_content_prefix = repo_content_prefix
        self.file_group = file_group
        self.code_renderer = RenderCode()
        self.tag_graphs = {} if cache_graphs else None

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        mentioned_entities=None,
        add_prefix: bool = True,
        search_terms: Set[str] | None = None,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()
        if not mentioned_entities:
            mentioned_entities = set()

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                self.max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                mentioned_entities,
                search_terms,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        num_tokens = self.token_count(files_listing)
        if self.verbose:
            self.io.tool_output(f"Repo-map: {num_tokens/1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix and add_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_tag_graph(self, abs_fnames: List[str] | None = None) -> TagGraph:
        if not abs_fnames:
            abs_fnames = self.file_group.get_all_filenames()
        clean_fnames = self.file_group.validate_fnames(abs_fnames)

        if self.tag_graphs is not None:
            for files, graph in self.tag_graphs.items():
                if not set(clean_fnames).difference(set(files)):
                    return graph
        # If no caching or cached graph not found, construct it
        all_tags = []
        code_map = {}
        for fname in clean_fnames:
            code, tags = self.tags_from_filename(fname)
            all_tags += tags
            code_map[fname] = code
        raw_graph = build_tag_graph(all_tags, code_map)
        graph = only_defs(raw_graph)
        if self.tag_graphs is not None:
            self.tag_graphs[tuple(clean_fnames)] = graph
        return graph

    def tags_from_filename(self, fname):
        def get_tags_raw_function(fname):
            code = read_text(fname)
            rel_fname = self.file_group.get_rel_fname(fname)
            data = get_tags_raw(fname, rel_fname, code)
            assert isinstance(data, list)
            return code, data

        # return get_tags_raw_function(fname)
        # # TODO: resume caching
        return self.file_group.cached_function_call(fname, get_tags_raw_function)

    def get_ranked_tags(
        self,
        chat_fnames,
        other_fnames,
        mentioned_fnames,
        mentioned_idents,
        mentioned_entities,
        search_terms,
    ):

        # Check file names for validity
        fnames = sorted(set(chat_fnames).union(set(other_fnames)))

        # Do better filename matching
        clean_mentioned_filenames = []
        for name in mentioned_fnames:
            for fname in fnames:
                if name in fname:
                    clean_mentioned_filenames.append(fname)
                    break
        mentioned_fnames = clean_mentioned_filenames

        # What does that do?
        if self.cache_missing:
            fnames = tqdm(fnames)
        self.cache_missing = False

        cleaned = self.file_group.validate_fnames(fnames)

        # All the source code parsing happens here
        tag_graph = self.get_tag_graph(cleaned)
        self.code_renderer.code_map = tag_graph.code_renderer.code_map

        tags = list(tag_graph.nodes)

        other_rel_fnames = [self.file_group.get_rel_fname(fname) for fname in other_fnames]
        if self.use_old_ranking:
            ranked_tags = rank_tags(
                tags, mentioned_fnames, mentioned_idents, chat_fnames, other_rel_fnames
            )
        else:
            ranked_tags = rank_tags_new(
                tag_graph,
                mentioned_fnames,
                mentioned_idents,
                mentioned_entities,
                chat_fnames,
                other_rel_fnames,
                search_terms,
            )

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        mentioned_entities=None,
        search_terms=None,
    ):
        """
        Does a binary search over the number of tags to include in the map,
        to find the largest map that fits within the token limit.
        :param chat_fnames:
        :param other_fnames:
        :param max_map_tokens:
        :param mentioned_fnames:
        :param mentioned_idents:
        :return:
        """
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()
        if not mentioned_entities:
            mentioned_entities = set()
        if not search_terms:
            search_terms = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            mentioned_entities,
            search_terms,
        )

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = [self.file_group.get_rel_fname(fname) for fname in chat_fnames]

        # Guess a small starting number to help with giant repos
        middle = min(max_map_tokens // 25, num_tags)

        self.tree_cache = dict()

        while lower_bound <= upper_bound:
            used_tags = [tag for tag in ranked_tags[:middle]]  # if tag[0] not in chat_rel_fnames]
            tree = self.code_renderer.to_tree(used_tags)
            num_tokens = self.token_count(tree)

            if num_tokens < max_map_tokens and num_tokens > best_tree_tokens:
                best_tree = tree
                best_tree_tokens = num_tokens

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    # tree_cache = dict()

    def repo_map_from_message(
        self,
        message: str,
        abs_added_fnames: Set[str] | None = None,
        rel_added_fnames: Set[str] | None = None,
        mentioned_entities: Set[str] | None = None,
        add_prefix: bool = False,
        llm=None,
    ) -> str:
        # TODO: this is a hack, should be able to supply either/or with one argument
        if rel_added_fnames is None:
            rel_added_fnames = set()
        else:
            rel_added_fnames = set([self.file_group.abs_root_path(f) for f in rel_added_fnames])

        if not abs_added_fnames:
            abs_added_fnames = set()

        abs_added_fnames = abs_added_fnames.union(rel_added_fnames)

        if llm is not None:
            search_terms = search_terms_from_message(message, llm)
        else:
            search_terms = set()

        cur_msg_text = message
        all_files = self.file_group.get_all_filenames()
        other_files = set(all_files) - set(abs_added_fnames)

        mentioned_fnames = self.file_group.get_file_mentions(cur_msg_text, abs_added_fnames)
        mentioned_idents = get_ident_mentions(cur_msg_text)

        all_rel_fnames = [self.file_group.get_rel_fname(f) for f in all_files]
        mentioned_fnames.update(get_ident_filename_matches(mentioned_idents, all_rel_fnames))

        repo_content = self.get_repo_map(
            abs_added_fnames,
            other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            mentioned_entities=mentioned_entities,
            add_prefix=add_prefix,
            search_terms=search_terms,
        )

        # fall back to global repo map if files in chat are disjoint from rest of repo
        if not repo_content:
            repo_content = self.get_repo_map(
                set(),
                set(all_files),
                mentioned_fnames=mentioned_fnames,
                mentioned_idents=mentioned_idents,
                add_prefix=add_prefix,
                search_terms=search_terms,
            )

        # fall back to completely unhinted repo
        if not repo_content:
            repo_content = self.get_repo_map(
                set(),
                set(all_files),
                add_prefix=add_prefix,
                search_terms=search_terms,
            )

        return repo_content


def search_terms_from_message(message: str, llm) -> Set[str]:
    search_prompt = f"""You are an expert bug fixer. You are given a bug report. 
        Return a JSON list of at most 10 strings extracted from the bug report, that should be used
        in a full-text search of the codebase to find the part of the code that needs to be modified. 
        Select at most 10 strings that are most likely to be unique to the part of the code that needs to be modified.
        ONLY extract strings that you could expect to find verbatim in the code, especially function names,
        class names, and error messages. 
        For method calls, such as `foo.bar()`, extract `.bar(` 

        For error messages, extract the bits of the error message that are likely to be found VERBATIM in the code, 
        for example "File not found: " rather than "File not found: /amger/gae/doc.tcx"; 
        return "A string is required" rather than "A string is required, not 'MyWeirdClassName'".

        Here is the problem description:
        {message}"""

    class ListOfStrings(BaseModel):
        strings: List[str] = Field(
            description="List of full-text search strings to find the part of the code that needs to be modified."
        )

    out = llm.with_structured_output(ListOfStrings).invoke(search_prompt)
    re_out = [x.split(".")[-1] for x in out.strings]
    re_out = sum([x.split(",") for x in re_out], [])
    return set(re_out)


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = ""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())
    for lang, ext in data:
        res += "<tr>"
        res += f'<td style="text-align: center;">{lang:20}</td>\n'
        res += f'<td style="text-align: center;">{ext:20}</td>\n'
        res += "</tr>"
    return res
