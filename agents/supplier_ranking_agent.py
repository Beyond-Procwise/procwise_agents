# ProcWise/agents/supplier_ranking_agent.py

import json
import pandas as pd
import ollama
import os
import logging
from .base_agent import BaseAgent

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class SupplierRankingAgent(BaseAgent):

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.prompt_library = self._load_json_file('prompts/SupplierRankingAgent_PromptLibrary.json')
        self.justification_template = self._load_json_file('prompts/Justification_Tag_Prompt_Template.json')
        self.policy_engine = agent_nick.policy_engine

    def run(self, intent: dict, data: pd.DataFrame) -> dict:
        logger.info("SupplierRankingAgent: Executing core logic...")
        criteria = intent.get('parameters', {}).get('criteria', [])

        weight_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'WeightAllocationPolicy'), {})
        weights = weight_policy.get('details', {}).get('rules', {}).get('default_weights', {})
        applicable_weights = {c: weights.get(c, 0) for c in criteria if weights.get(c, 0) > 0}

        # --- UPGRADE: Expand Ranking Criteria ---
        # 1. Apply categorical scoring first for criteria like 'payment_terms'
        scored_df = self._score_categorical_criteria(data, applicable_weights.keys())

        # 2. Normalize the remaining numeric scores
        norm_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'NormalizationDirectionPolicy'), {})
        criteria_direction = norm_policy.get('details', {}).get('rules', {})
        scored_df = self._normalize_numeric_scores(scored_df, criteria_direction)

        # 3. Calculate the final weighted score
        scored_df['final_score'] = 0.0
        for criterion, weight in applicable_weights.items():
            score_col = f"{criterion}_score"
            if score_col in scored_df.columns:
                scored_df['final_score'] += scored_df[score_col].fillna(0) * weight
            else:
                logger.warning(f"Score column '{score_col}' not found for weighting.")

        ranked_df = scored_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        logger.info("<- Ranking complete.")

        logger.info("-> Generating justifications for top suppliers...")
        ranked_df['justification'] = ''
        for i, row in ranked_df.head(3).iterrows():
            ranked_df.at[i, 'justification'] = self._generate_justification(row, applicable_weights.keys())

        return json.loads(ranked_df.to_json(orient='records'))

    def _load_json_file(self, relative_path: str) -> dict:
        absolute_path = os.path.join(PROJECT_ROOT, relative_path)
        try:
            with open(absolute_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.critical(
                f"FATAL ERROR (SupplierRankingAgent): Failed to load/parse JSON from {absolute_path}. Error: {e}")
            raise

    def _score_categorical_criteria(self, df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        scored_df = df.copy()
        cat_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'CategoricalScoringPolicy'), None)
        if not cat_policy: return scored_df

        policy_rules = cat_policy.get('details', {}).get('rules', {})
        for criterion in [c for c in criteria if c in policy_rules]:
            logger.info(f"Applying categorical scoring for '{criterion}'")
            raw_col = f"{criterion}_raw"
            score_col = f"{criterion}_score"
            if raw_col in scored_df.columns:
                mapping = policy_rules[criterion]
                default_value = mapping.get('default', 0)
                scored_df[score_col] = scored_df[raw_col].map(mapping).fillna(default_value)
        return scored_df

    def _normalize_numeric_scores(self, df: pd.DataFrame, criteria_map: dict) -> pd.DataFrame:
        normalized_df = df.copy()
        for criterion, direction in criteria_map.items():
            raw_col = f"{criterion}_score_raw"
            score_col = f"{criterion}_score"

            if score_col in normalized_df.columns or raw_col not in df.columns: continue

            min_val, max_val = df[raw_col].min(), df[raw_col].max()
            if (max_val - min_val) == 0: normalized_df[score_col] = 10.0; continue

            if direction == 'lower_is_better':
                normalized_df[score_col] = 10 * (max_val - df[raw_col]) / (max_val - min_val)
            else:  # higher_is_better
                normalized_df[score_col] = 10 * (df[raw_col] - min_val) / (max_val - min_val)

        return normalized_df

    def _generate_justification(self, supplier_row, criteria) -> str:
        if not self.justification_template: return "Justification prompt template not found."

        score_details = []
        for c in criteria:
            score_col = f"{c}_score"
            if score_col in supplier_row and pd.notna(supplier_row[score_col]):
                score_details.append(f"- {c.replace('_', ' ').title()}: {supplier_row[score_col]:.2f}")

        prompt = self.justification_template['prompt_template'].format(
            supplier_name=supplier_row.get('supplier_name', 'N/A'),
            final_score=supplier_row.get('final_score', 0),
            score_breakdown="\n".join(score_details)
        )
        try:
            response = ollama.generate(model=self.settings.extraction_model, prompt=prompt, stream=False)
            return response.get('response', 'Could not generate justification.').strip()
        except Exception as e:
            logger.error(f"ERROR generating justification: {e}");
            return "Justification generation failed."

