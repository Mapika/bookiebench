# Name check

**Decision (2026-09-27): the project is named BookieBench.** On the same date, the PyPI name `bookiebench`
was free (`https://pypi.org/pypi/bookiebench/json` returned 404), a GitHub repository search for "bookiebench"
returned 0 results, and a Hugging Face dataset search returned none. The check of the internal working name follows.

## Former working name "HonestBench" (checked 2026-09-27)

This is a web search for existing projects using the name. 

| where | project | what it is | conflict? |
|---|---|---|---|
| GitHub | [empire-mind/honestbench](https://github.com/empire-mind/honestbench) | an eval harness that audits agent trajectories for verification ("lucky-pass detection"); MIT, 0 stars, 2 commits, last push 2026-09-26 | **Same name, AI-evaluation domain.** The closest conflict |
| GitHub | nickharris808/honestbench | "Your CI is green. Delete the evidence it checks — is it still green?"; 0 stars, last push 2026-09-04 | same name, CI-testing domain |
| GitHub / Glama | [maymun207/mcp-honestbench](https://glama.ai/mcp/servers/maymun207/mcp-honestbench) | a deliberately dishonest MCP server for testing whether agents detect falsehoods | similar name, agent-honesty domain |
| arXiv | [HonestyBench, arXiv:2510.17509](https://arxiv.org/pdf/2510.17509) ("Annotation-Efficient Universal Honesty Alignment") | a large benchmark of freeform factual QA datasets for LLM honesty/confidence alignment | near-identical name (one letter), same broad field (LLM confidence). **Most likely to be confused in citations** |
| related | BeHonest (GAIR-NLP), HonestLLM / HoneSet | LLM honesty benchmarks | different names, overlapping theme |
| PyPI | `honestbench` | `https://pypi.org/pypi/honestbench/json` returned 404: not taken | none |
| Hugging Face | datasets / models search "honestbench" | empty results | none |

Summary: the name is free on PyPI and Hugging Face, but three small GitHub projects already use "honestbench". One
of them is an AI-agent eval harness. A published paper benchmark is called "HonestyBench". The "honesty" framing
also invites confusion with LLM-honesty (truthfulness/deception) benchmarks, which measure something different from
probabilistic coherence.
