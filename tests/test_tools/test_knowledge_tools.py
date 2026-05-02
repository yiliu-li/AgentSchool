"""Tests for research, book, knowledge graph, and transcript tools."""

from __future__ import annotations

import httpx
import pytest

from agentschool.tools import create_default_tool_registry
from agentschool.tools.base import ToolExecutionContext
from agentschool.tools.knowledge_tools import (
    ArxivSearchTool,
    ResearchSwarmInput,
    ResearchSwarmTool,
    SearchInput,
    YouTubeTranscriptInput,
    YouTubeTranscriptTool,
)


def test_knowledge_tools_are_registered() -> None:
    registry = create_default_tool_registry()
    names = {tool.name for tool in registry.list_tools()}

    assert "research_arxiv_search" in names
    assert "research_semantic_scholar_search" in names
    assert "research_openalex_search" in names
    assert "research_crossref_search" in names
    assert "research_pubmed_search" in names
    assert "books_openlibrary_search" in names
    assert "books_gutendex_search" in names
    assert "wikidata_sparql" in names
    assert "web_reader" in names
    assert "video_youtube_transcript" in names
    assert "research_swarm" in names


@pytest.mark.asyncio
async def test_arxiv_search_parses_atom(tmp_path, monkeypatch) -> None:
    async def fake_fetch(url: str, **_: object) -> httpx.Response:
        request = httpx.Request("GET", url)
        body = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/1234.5678</id>
            <published>2026-01-02T00:00:00Z</published>
            <title>Useful Agent Skills</title>
            <summary>Skills help agents solve tasks.</summary>
            <author><name>Ada Lovelace</name></author>
          </entry>
        </feed>"""
        return httpx.Response(200, text=body, request=request)

    monkeypatch.setitem(ArxivSearchTool.execute.__globals__, "fetch_public_http_response", fake_fetch)

    result = await ArxivSearchTool().execute(
        SearchInput(query="agent skills", max_results=1),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Useful Agent Skills" in result.output
    assert "Ada Lovelace" in result.output


@pytest.mark.asyncio
async def test_youtube_transcript_parses_caption_xml(tmp_path, monkeypatch) -> None:
    async def fake_fetch(url: str, **_: object) -> httpx.Response:
        request = httpx.Request("GET", url)
        body = '<transcript><text start="1.2">hello &amp; welcome</text></transcript>'
        return httpx.Response(200, text=body, request=request)

    monkeypatch.setitem(YouTubeTranscriptTool.execute.__globals__, "fetch_public_http_response", fake_fetch)

    result = await YouTubeTranscriptTool().execute(
        YouTubeTranscriptInput(url_or_id="https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert "dQw4w9WgXcQ" in result.output
    assert "[1.2s] hello & welcome" in result.output


@pytest.mark.asyncio
async def test_research_swarm_runs_selected_sources(tmp_path, monkeypatch) -> None:
    async def fake_arxiv(query: str, max_results: int) -> str:
        return f"arxiv {query} {max_results}"

    async def fake_openlibrary(query: str, max_results: int) -> str:
        return f"openlibrary {query} {max_results}"

    monkeypatch.setitem(ResearchSwarmTool.execute.__globals__, "_search_arxiv", fake_arxiv)
    monkeypatch.setitem(ResearchSwarmTool.execute.__globals__, "_search_openlibrary", fake_openlibrary)

    result = await ResearchSwarmTool().execute(
        ResearchSwarmInput(
            query="agent skills",
            sources=["arxiv", "openlibrary"],
            max_results_per_source=2,
        ),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Sources: arxiv, openlibrary" in result.output
    assert "arxiv agent skills 2" in result.output
    assert "openlibrary agent skills 2" in result.output
