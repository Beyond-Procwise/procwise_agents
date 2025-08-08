# ProcWise/agents/supplier_ranking_agent.py

import json
import os
import logging
import pandas as pd
import ollama

from .base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class SupplierRankingAgent(BaseAgent):
    """
    Ranks suppliers based on provided criteria and data from QueryEngine.
    Expects `supplier_data` in context.input_data as list of dicts or DataFrame.
    """

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.prompt_library = self._load_json_file('prompts/SupplierRankingAgent_PromptLibrary.json')
        self.justification_template = self._load_json_file('prompts/Justification_Tag_Prompt_Template.json')
        self.policy_engine = agent_nick.policy_engine

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("SupplierRankingAgent: Starting ranking...")

        # 1. Load supplier_data from context
        supplier_data = context.input_data.get('supplier_data')
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
                logger.error(f"Invalid supplier_data format: {e}")
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
                scored_df['final_score'] += scored_df[col].fillna(0) * w
            else:
                logger.warning(f"Criterion column missing: {col}")

        ranked_df = scored_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

        # 6. Generate justifications for top-N
        top_n = intent.get('parameters', {}).get('top_n', 3)
        ranked_df['justification'] = ranked_df.apply(
            lambda row: self._generate_justification(row, weights.keys()) if row.name < top_n else '',
            axis=1
        )

        ranking = json.loads(ranked_df.to_json(orient='records'))
        logger.info("SupplierRankingAgent: Ranking complete.")
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={'ranking': ranking}
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
                out[score_col] = (
                    10 * (max_v - vals) / (max_v - min_v)
                    if direction == 'lower_is_better'
                    else 10 * (vals - min_v) / (max_v - min_v)
                )
        return out

    def _generate_justification(self, row: pd.Series, criteria: list) -> str:
        if not self.justification_template:
            return "No justification template available."
        breakdown = []
        for crit in criteria:
            score_col = f"{crit}_score"
            if score_col in row:
                breakdown.append(f"- {crit.replace('_', ' ').title()}: {row[score_col]:.2f}")
        prompt = self.justification_template['prompt_template'].format(
            supplier_name=row.get('supplier_name', 'Unknown'),
            final_score=row.get('final_score', 0.0),
            score_breakdown="\n".join(breakdown)
        )
        try:
            resp = ollama.generate(
                model=self.settings.extraction_model,
                prompt=prompt,
                stream=False
            )
            return resp.get('response', '').strip()
        except Exception as e:
            logger.error(f"Justification generation failed: {e}")
            return "Justification generation failed."
