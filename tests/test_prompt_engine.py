import json

from orchestration.prompt_engine import PromptEngine


class DummyCursor:
    def __init__(self, rows):
        self._rows = rows
        self.description = [
            ("prompt_id", None, None, None, None, None, None),
            ("prompt_name", None, None, None, None, None, None),
            ("prompt_type", None, None, None, None, None, None),
            ("prompt_linked_agents", None, None, None, None, None, None),
            ("prompts_desc", None, None, None, None, None, None),
            ("prompts_status", None, None, None, None, None, None),
            ("created_date", None, None, None, None, None, None),
            ("created_by", None, None, None, None, None, None),
            ("last_modified_date", None, None, None, None, None, None),
            ("last_modified_by", None, None, None, None, None, None),
            ("version", None, None, None, None, None, None),
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, query, params=None):
        self.query = query

    def fetchall(self):
        return self._rows


class DummyConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor(self._rows)


def make_prompt_engine(rows):
    def get_conn():
        return DummyConn(rows)

    return PromptEngine(connection_factory=get_conn)


def test_prompt_engine_loads_and_normalises_prompts():
    prompt_payload = {
        "templates": [
            {
                "template_id": "contract",
                "description": "Detect expiring contracts",
            }
        ],
        "prompt_template": "Use policy {{policy_id}}",
        "metadata": {"workflow": "contract_expiry_check"},
        "prompt_config": {"negotiation_window_days": 90},
    }

    prompt_rows = [
        (
            42,
            "Opportunity Prompt",
            "json",
            "{Opportunity_Mining}",
            json.dumps(prompt_payload),
            1,
            None,
            None,
            None,
            None,
            None,
        )
    ]

    engine = make_prompt_engine(prompt_rows)
    catalog = engine.prompts_by_id()
    assert 42 in catalog

    prompt = catalog[42]
    assert prompt["promptId"] == 42
    assert prompt["template"] == "Use policy {{policy_id}}"
    assert prompt["metadata"] == {"workflow": "contract_expiry_check"}
    assert prompt["prompt_config"] == {"negotiation_window_days": 90}
    assert prompt["linked_agents"] == ["opportunity_mining"]

    agent_prompts = engine.prompts_for_agent("opportunity_mining")
    assert agent_prompts and agent_prompts[0]["promptId"] == 42

    library = engine.prompt_library()
    assert any(t.get("template_id") == "contract" for t in library["templates"])
