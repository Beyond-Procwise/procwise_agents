# ProcWise/engines/policy_engine.py

import json
import os
from .base_engine import BaseEngine


class PolicyEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.supplier_policies = self._load_json_file('policies/Supplier_Ranking_Policies.json')
        self.opportunity_policies = self._load_json_file('policies/Opportunity_Policies.json')
        self._normalise_weight_policy()
        print("PolicyEngine: All policies loaded.")

    def _normalise_weight_policy(self) -> None:
        """Ensure default weights sum to 1."""
        weight_policy = next(
            (p for p in self.supplier_policies if p.get('policyName') == 'WeightAllocationPolicy'),
            None,
        )
        if not weight_policy:
            return
        weights = weight_policy.get('details', {}).get('rules', {}).get('default_weights', {})
        total = sum(weights.values())
        if total and abs(total - 1.0) > 1e-6:
            for key in list(weights.keys()):
                weights[key] = round(weights[key] / total, 4)

    def validate_workflow(self, workflow_name: str, user_id: str, input_data: dict) -> dict:
        """Validate a workflow against policy rules.

        For supplier ranking workflows we build a ranking intent and
        delegate validation to :meth:`validate_and_apply`.  The tuple
        returned by that method is then converted into a simple dict with
        ``allowed`` and ``reason`` fields so that upstream callers have a
        consistent interface.  All other workflows currently bypass
        policy checks and are allowed by default.
        """
        if workflow_name == 'supplier_ranking':
            criteria = input_data.get('criteria')
            if not criteria:
                weight_policy = next(
                    (p for p in self.supplier_policies if p.get('policyName') == 'WeightAllocationPolicy'),
                    {},
                )
                criteria = list(
                    weight_policy.get('details', {}).get('rules', {}).get('default_weights', {}).keys()
                )
            intent = {
                'template_id': 'rank_by_criteria',
                'parameters': {
                    'criteria': criteria
                }
            }
            allowed, reason, _ = self.validate_and_apply(intent)
            return {'allowed': allowed, 'reason': reason}

        return {'allowed': True, 'reason': 'No policy checks'}

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
