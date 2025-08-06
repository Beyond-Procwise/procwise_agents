# ProcWise/agents/supplier_ranking_agent.py

import json
import pandas as pd
import ollama
import os
from .base_agent import BaseAgent

# --- Define the project's root directory dynamically ---
# This ensures that configuration files are found regardless of how the script is run.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class SupplierRankingAgent(BaseAgent):
    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        # The agent needs its own prompt library for the orchestrator to use
        self.prompt_library = self._load_json_file('prompts/SupplierRankingAgent_PromptLibrary.json')
        self.justification_template = self._load_json_file('prompts/Justification_Tag_Prompt_Template.json')
        # Access the already loaded policies via the orchestrator's policy engine
        self.policy_engine = agent_nick.policy_engine

    def run(self, intent: dict, data: pd.DataFrame) -> dict:
        """
        The agent's core logic: applies scoring, ranking, and justification to pre-fetched data.
        This method is called by the Orchestrator.
        """
        print("SupplierRankingAgent: Executing core logic...")

        # 1. Get applicable criteria and weights from the validated intent and policies
        criteria = intent.get('parameters', {}).get('criteria', [])
        weight_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'WeightAllocationPolicy'), {})
        weights = weight_policy.get('details', {}).get('rules', {}).get('default_weights', {})
        applicable_weights = {c: weights.get(c, 0) for c in criteria}

        # 2. Get normalization rules from policies
        norm_policy = next(
            (p for p in self.policy_engine.supplier_policies if p['policyName'] == 'NormalizationDirectionPolicy'), {})
        criteria_direction = norm_policy.get('details', {}).get('rules', {})

        # 3. Normalize raw scores into a 0-10 scale
        scored_df = self._normalize_scores(data, criteria_direction)

        # 4. Calculate the final weighted score
        scored_df['final_score'] = 0.0
        for criterion, weight in applicable_weights.items():
            scored_df['final_score'] += scored_df.get(f"{criterion}_score", 0) * weight

        # 5. Rank the suppliers
        ranked_df = scored_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        print("<- Ranking complete.")

        # 6. Generate Justification for top 3 suppliers
        print("-> Generating justifications for top suppliers...")
        ranked_df['justification'] = ''  # Create an empty column
        for i, row in ranked_df.head(3).iterrows():  # Generate for top 3
            ranked_df.at[i, 'justification'] = self._generate_justification(row)

        return json.loads(ranked_df.to_json(orient='records'))

    def _load_json_file(self, relative_path: str) -> dict:
        """Loads a JSON file using an absolute path from the project root."""
        absolute_path = os.path.join(PROJECT_ROOT, relative_path)
        try:
            with open(absolute_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR (SupplierRankingAgent): Could not find file at: {absolute_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR (SupplierRankingAgent): Failed to load/parse JSON from {absolute_path}. Error: {e}")
            raise

    def _normalize_scores(self, df: pd.DataFrame, criteria_map: dict) -> pd.DataFrame:
        """Normalizes raw data into a 0-10 scale based on scoring logic."""
        normalized_df = df.copy()
        for criterion, direction in criteria_map.items():
            raw_col = f"{criterion}_score_raw"
            if raw_col not in df.columns: continue

            min_val, max_val = df[raw_col].min(), df[raw_col].max()

            # Avoid division by zero if all values in the column are the same
            if (max_val - min_val) == 0:
                normalized_df[f"{criterion}_score"] = 10.0
                continue

            if direction == 'lower_is_better':
                # Inverse Min-Max Normalization: Higher score for lower values
                normalized_df[f"{criterion}_score"] = 10 * (max_val - df[raw_col]) / (max_val - min_val)
            else:  # higher_is_better
                # Standard Min-Max Normalization: Higher score for higher values
                normalized_df[f"{criterion}_score"] = 10 * (df[raw_col] - min_val) / (max_val - min_val)
        return normalized_df

    def _generate_justification(self, supplier_row) -> str:
        """Uses an LLM to generate a justification tag for a single supplier."""
        if not self.justification_template:
            return "Justification prompt template not found."

        # Dynamically build the context string from available scores
        score_details = []
        for score_key in ['price_score', 'delivery_score', 'risk_score']:
            if score_key in supplier_row and pd.notna(supplier_row[score_key]):
                score_details.append(f"- {score_key.replace('_', ' ').title()}: {supplier_row[score_key]:.2f}")

        prompt = self.justification_template['prompt_template'].format(
            supplier_name=supplier_row.get('supplier_name', 'N/A'),
            final_score=supplier_row.get('final_score', 0),
            score_breakdown="\n".join(score_details)
        )

        try:
            response = ollama.generate(
                model=self.settings.extraction_model,
                prompt=prompt,
                stream=False
            )
            return response.get('response', 'Could not generate justification.').strip()
        except Exception as e:
            print(f"ERROR generating justification: {e}")
            return "Justification generation failed."

