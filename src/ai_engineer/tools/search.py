# Copyright 2026 Matt Dyer / Dyer-Tech
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web and paper search."""
from __future__ import annotations

from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="web_search",
    description="Search the web for a query and return top results with snippets.",
)
def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise ToolError("Install duckduckgo-search: pip install duckduckgo-search") from e
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return "\n\n".join(f"### {r['title']}\n{r['href']}\n{r['body']}" for r in results)


@tool(
    name="arxiv_search",
    description="Search arXiv for papers by query. Returns title, authors, abstract, link.",
)
def arxiv_search(query: str, max_results: int = 5) -> str:
    try:
        import arxiv
    except ImportError as e:
        raise ToolError("Install arxiv: pip install arxiv") from e
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    client = arxiv.Client()
    out = []
    for r in client.results(search):
        out.append(f"### {r.title}\n{r.entry_id}\n{' '.join(a.name for a in r.authors)}\n\n{r.summary[:500]}")
    return "\n\n---\n\n".join(out)


@tool(
    name="fetch_url",
    description="Fetch a URL and return its text content.",
)
def fetch_url(url: str, max_chars: int = 50_000) -> str:
    import httpx

    r = httpx.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    text = r.text
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text
