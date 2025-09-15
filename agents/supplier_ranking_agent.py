# ProcWise/agents/supplier_ranking_agent.py

import json
import os
import logging
import re
import pandas as pd
import ollama

from utils.gpu import configure_gpu
from .base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

configure_gpu()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class SupplierRankingAgent(BaseAgent):
    """Rank suppliers based on criteria and supplier data.

    The agent can operate standalone when given appropriate prompts that
    include supplier information.  However, in a full workflow it is typically
    invoked *after* the Opportunity Miner so that the ranked supplier list
    aligns with candidates identified for a particular opportunity.  Running it
    earlier may yield rankings for suppliers unrelated to the downstream
    opportunity context.

    Expects ``supplier_data`` in ``context.input_data`` as a list of dicts or a
    ``DataFrame``.  If ``supplier_data`` is omitted, the agent will attempt to
    fetch it via the ``QueryEngine``.  By default, the top three suppliers are
    returned.
    """

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.prompt_library = self._load_json_file('prompts/SupplierRankingAgent_PromptLibrary.json')
        self.justification_template = self._load_json_file('prompts/Justification_Tag_Prompt_Template.json')
        self.policy_engine = agent_nick.policy_engine
        # Allow the agent to fetch its own supplier data when invoked outside
        # the standard workflow.
        self.query_engine = agent_nick.query_engine

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("SupplierRankingAgent: Starting ranking...")

        if hasattr(self.query_engine, "train_procurement_context"):
            try:
                self.query_engine.train_procurement_context()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to train procurement context")

        # 1. Load supplier_data from context or fetch dynamically
        supplier_data = context.input_data.get('supplier_data')
        if supplier_data is None:
            try:
                supplier_data = self.query_engine.fetch_supplier_data(
                    context.input_data
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to fetch supplier data")
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="Failed to fetch supplier data",
                )
            if supplier_data is None:
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error='No supplier_data provided to SupplierRankingAgent'
                )
        # Convert to DataFrame
        if isinstance(supplier_data, pd.DataFrame):
            df = supplier_data.copy()
        else:
            try:
                df = pd.DataFrame(supplier_data)
            except Exception as e:
                logger.exception("Invalid supplier_data format")
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error='Failed to parse supplier_data into DataFrame'
                )
        if df.empty:
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error='supplier_data is empty'
            )
        candidate_ids = context.input_data.get("supplier_candidates")
        if candidate_ids:
            # normalize candidate_ids to a list
            if not isinstance(candidate_ids, (list, set, tuple)):
                try:
                    candidate_ids = list(candidate_ids)
                except Exception:
                    candidate_ids = [candidate_ids]
            candidate_set = set(candidate_ids)
            # filter DataFrame to only candidate suppliers
            if "supplier_id" not in df.columns:
                logger.error("supplier_data does not contain 'supplier_id' column.")
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="supplier_data missing 'supplier_id' column",
                )
            df = df[df["supplier_id"].isin(candidate_set)]
            if df.empty:
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="No matching suppliers found for candidates",
                )
            logger.debug("SupplierRankingAgent: filtered candidates %s", list(candidate_set))
        # 2. Determine criteria and weights
        intent = context.input_data.get('intent', {})
        requested = intent.get('parameters', {}).get('criteria', [])
        weight_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'WeightAllocationPolicy'),
            {}
        )
        default_weights = weight_policy.get('details', {}).get('rules', {}).get('default_weights', {})
        criteria = requested if requested else list(default_weights.keys())
        weights = {c: default_weights.get(c, 0) for c in criteria if default_weights.get(c, 0) > 0}

        # 3. Apply categorical scoring
        scored_df = self._score_categorical_criteria(df, weights.keys())

        # 4. Apply numeric normalization
        norm_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'NormalizationDirectionPolicy'),
            {}
        )
        direction_map = norm_policy.get('details', {}).get('rules', {})
        scored_df = self._normalize_numeric_scores(scored_df, direction_map)

        # 5. Calculate final weighted score
        scored_df['final_score'] = 0.0
        for crit, w in weights.items():
            col = f"{crit}_score"
            if col in scored_df.columns:
                scored_df['final_score'] += pd.to_numeric(scored_df[col].fillna(0), errors='coerce') * float(w)

            else:
                logger.warning(f"Criterion column missing: {col}")

        ranked_df = scored_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

        # 6. Generate justifications for top-N and limit output
        top_n = intent.get('parameters', {}).get('top_n')
        if not top_n:
            query_text = context.input_data.get('query', '')
            match = re.search(r'top[-\s]*(\d+)', query_text, re.IGNORECASE)
            top_n = int(match.group(1)) if match else 3
        top_n = max(1, min(int(top_n), len(ranked_df)))

        top_df = ranked_df.head(top_n).copy()
        top_df['justification'] = top_df.apply(
            lambda row: self._generate_justification(row, weights.keys()),
            axis=1
        )

        ranking = json.loads(top_df.to_json(orient='records'))
        logger.info("SupplierRankingAgent: Ranking complete with %d entries", len(ranking))
        logger.debug("SupplierRankingAgent ranking: %s", ranking)
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={'ranking': ranking},
            pass_fields={'ranking': ranking},
        )

    def _load_json_file(self, rel_path: str) -> dict:
        path = os.path.join(PROJECT_ROOT, rel_path)
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.critical(f"Failed to load JSON at {path}: {e}")
            raise

    def _score_categorical_criteria(self, df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        out = df.copy()
        policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'CategoricalScoringPolicy'),
            None
        )
        if not policy:
            return out
        rules = policy.get('details', {}).get('rules', {})
        for crit in criteria:
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col in out.columns and crit in rules:
                mapping = rules[crit]
                out[score_col] = out[raw_col].map(mapping).fillna(mapping.get('default', 0))
        return out

    def _normalize_numeric_scores(self, df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
        out = df.copy()
        for crit, direction in dirs.items():
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col not in df.columns:
                continue
            vals = df[raw_col]
            min_v, max_v = vals.min(), vals.max()
            if max_v - min_v == 0:
                out[score_col] = 10.0
            else:
                range_diff = float(max_v - min_v)  # Convert to float to avoid integer division issues
                if direction == 'lower_is_better':
                    out[score_col] = 10 * (max_v - vals) / range_diff
                else:
                    out[score_col] = 10 * (vals - min_v) / range_diff
        return out

    def _generate_justification(self, row: pd.Series, criteria: list) -> str:
        if not self.justification_template:
            return "No justification template available."
        breakdown = []
        for crit in criteria:
            score_col = f"{crit}_score"
            if score_col in row:
                score_value = row.get(score_col)
                if isinstance(score_value, (int, float)):
                    breakdown.append(f"- {crit.replace('_', ' ').title()}: {score_value:.2f}")
                else:
                    logger.warning(f"Score value for {crit} is not numeric: {score_value}")
                    breakdown.append(f"- {crit.replace('_', ' ').title()}: N/A")
        prompt = self.justification_template['prompt_template'].format(
            supplier_name=row.get('supplier_name', 'Unknown'),
            final_score=row.get('final_score', 0.0),
            score_breakdown="\n".join(breakdown)
        )
        try:
            resp = self.call_ollama(prompt, model=self.settings.extraction_model)
            return resp.get('response', '').strip()
        except Exception as e:
            logger.exception("Justification generation failed")
            return "Justification generation failed."
