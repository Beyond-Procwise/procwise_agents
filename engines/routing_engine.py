import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fallback routing configuration used if no external model is provided.
routing_model: Dict[str, Any] = {
    "global_settings": {"max_chain_depth": 10},
    "routing_rules": [],
}


@dataclass
class RoutingDecision:
    """Routing decision result"""
    next_agents: List[str]
    pass_fields: Dict[str, Any]
    confidence: float
    reason: str


class RoutingEngine:
    """Engine for dynamic agent routing based on output conditions"""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.load_routing_rules()

    def load_routing_rules(self):
        """Load routing rules from configuration"""
        try:
            with open('models/routing_model.json', 'r') as f:
                self.routing_model = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load routing model: {e}")
            self.routing_model = routing_model  # Use default from code

    def evaluate_routing(self, agent_name: str, output: Any,
                         context: Any) -> Optional[Dict]:
        """Evaluate routing rules for agent output"""
        try:
            # Find rules for this agent
            rules = self._find_agent_rules(agent_name)
            if not rules:
                return None

            # Evaluate conditions
            for rule in rules:
                decision = self._evaluate_rule(rule, output, context)
                if decision:
                    # Log routing decision
                    logger.info(
                        f"Routing from {agent_name}: {decision.next_agents} "
                        f"(reason: {decision.reason})"
                    )

                    # Apply learning if enabled
                    if self.settings.enable_learning:
                        self._update_routing_weights(agent_name, decision)

                    return {
                        'next_agents': decision.next_agents,
                        'pass_fields': decision.pass_fields,
                        'metadata': {
                            'confidence': decision.confidence,
                            'reason': decision.reason
                        }
                    }

            return None

        except Exception as e:
            logger.error(f"Routing evaluation failed: {e}")
            return None

    def _find_agent_rules(self, agent_name: str) -> List[Dict]:
        """Find routing rules for agent"""
        rules = []

        for rule in self.routing_model.get('routing_rules', []):
            trigger = rule.get('trigger_agent', '')
            if trigger in agent_name or agent_name in trigger:
                rules.append(rule)

        return rules

    def _evaluate_rule(self, rule: Dict, output: Any, context: Any) -> Optional[RoutingDecision]:
        """Evaluate single routing rule"""
        conditions = rule.get('conditions', [])

        for condition in conditions:
            if self._check_condition(condition, output):
                return RoutingDecision(
                    next_agents=condition.get('next_agents', []),
                    pass_fields=self._extract_fields(
                        output,
                        condition.get('pass_fields', [])
                    ),
                    confidence=self._calculate_confidence(output),
                    reason=f"Condition matched: {condition.get('field', 'expression')}"
                )

        # Check failure condition
        if hasattr(output, 'status') and output.status.value == 'failed':
            failure_config = rule.get('on_failure', {})
            if failure_config:
                return RoutingDecision(
                    next_agents=[failure_config.get('fallback_agent', 'HumanEscalationAgent')],
                    pass_fields={'error': getattr(output, 'error', 'Unknown error')},
                    confidence=0.5,
                    reason='Failure condition triggered'
                )

        return None

    def _check_condition(self, condition: Dict, output: Any) -> bool:
        """Check if condition is met"""
        try:
            # Handle expression-based conditions
            if 'expression' in condition:
                # Create evaluation context
                eval_context = {
                    'output': output.data if hasattr(output, 'data') else output
                }
                return eval(condition['expression'], {"__builtins__": {}}, eval_context)

            # Handle field-based conditions
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')

            if not field:
                return False

            # Get field value from output
            if hasattr(output, 'data'):
                field_value = output.data.get(field)
            else:
                field_value = output.get(field) if isinstance(output, dict) else None

            # Evaluate operator
            if operator == '==':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            elif operator == '>':
                return field_value > value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<':
                return field_value < value
            elif operator == '<=':
                return field_value <= value
            elif operator == 'in':
                return field_value in value
            elif operator == 'contains':
                return value in str(field_value)

            return False

        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    def _extract_fields(self, output: Any, fields: List[str]) -> Dict:
        """Extract specified fields from output"""
        extracted = {}

        data = output.data if hasattr(output, 'data') else output
        if isinstance(data, dict):
            for field in fields:
                if field in data:
                    extracted[field] = data[field]

        return extracted

    def _calculate_confidence(self, output: Any) -> float:
        """Calculate routing confidence"""
        if hasattr(output, 'confidence'):
            return output.confidence
        return 0.8  # Default confidence

    def _update_routing_weights(self, agent_name: str, decision: RoutingDecision):
        """Update routing weights based on learning (placeholder)"""
        # This would connect to a learning system to improve routing over time
        pass
