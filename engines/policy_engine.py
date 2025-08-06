# ProcWise/engines/policy_engine.py

import json
import os
from .base_engine import BaseEngine


class PolicyEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.supplier_policies = self._load_json_file('policies/Supplier_Ranking_Policies.json')
        self.opportunity_policies = self._load_json_file('policies/Opportunity_Policies.json')
        print("PolicyEngine: All policies loaded.")

    def validate_and_apply(self, intent: dict) -> tuple[bool, str, dict]:
        # ... (rest of the method is the same)
        print(f"PolicyEngine: Validating intent -> {intent}")
        if not intent or not intent.get('parameters'):
            return False, "Query intent could not be determined.", intent

        if intent.get('template_id') == 'rank_by_criteria':
            criteria = intent.get('parameters', {}).get('criteria')
            if not criteria or not isinstance(criteria, list) or len(criteria) == 0:
                reason = "Policy validation failed: Ranking requires at least one criterion (e.g., price, delivery)."
                return False, reason, intent

            weight_policy = next((p for p in self.supplier_policies if p['policyName'] == 'WeightAllocationPolicy'),
                                 None)
            if not weight_policy: return False, "Weight allocation policy not found.", intent

            defined_weights = weight_policy['details']['rules']['default_weights'].keys()
            for c in criteria:
                if c not in defined_weights:
                    reason = f"Policy validation failed: Ranking criterion '{c}' has no defined weight."
                    return False, reason, intent

        print("PolicyEngine: Intent validated successfully.")
        return True, "Validation successful.", intent
