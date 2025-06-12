"""
Risk fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate the risk profile
of access control policies.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    PolicyEffect,
    ConditionOperator
)
from src.darwin.fitness.base import FitnessFunction, FitnessMetrics
from src.darwin.fitness.metrics import RiskMetrics


class RiskFitness(FitnessFunction):
    """
    Fitness function that evaluates risk aspects of a policy.

    This function considers:
    - Insider threat risk
    - Data exposure risk
    - Privilege escalation risk
    - Anomaly detection capabilities
    - Incident response readiness
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk fitness function.

        Args:
            config: Configuration with keys:
                - insider_threat_weight: Weight for insider threat risk
                - data_exposure_weight: Weight for data exposure risk
                - privilege_escalation_weight: Weight for privilege escalation risk
                - high_risk_resources: List of high-risk resource patterns
                - sensitive_actions: List of sensitive action types
        """
        super().__init__(config)

        # Weights for different risk aspects
        self.weights = {
            'insider_threat': self.config.get('insider_threat_weight', 0.3),
            'data_exposure': self.config.get('data_exposure_weight', 0.3),
            'privilege_escalation': self.config.get('privilege_escalation_weight', 0.2),
            'anomaly_detection': self.config.get('anomaly_detection_weight', 0.1),
            'incident_response': self.config.get('incident_response_weight', 0.1)
        }

        # High-risk resources
        self.high_risk_resources = self.config.get('high_risk_resources', [
            'admin:*', 'iam:*', 'security:*', 'audit:*',
            '*:credentials:*', '*:secrets:*', '*:keys:*'
        ])

        # Sensitive actions
        self.sensitive_actions = self.config.get('sensitive_actions', [
            'delete', 'modify', 'grant', 'revoke', 'approve',
            'transfer', 'export', 'download'
        ])

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the risk fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Risk fitness score between 0 and 1 (higher = lower risk)
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.overall_risk_score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> RiskMetrics:
        """
        Calculate detailed risk metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed risk metrics
        """
        # Calculate individual risk scores (inverted - higher score = lower risk)
        insider_score = 1.0 - self._calculate_insider_threat_risk(chromosome)
        exposure_score = 1.0 - self._calculate_data_exposure_risk(chromosome)
        escalation_score = 1.0 - self._calculate_privilege_escalation_risk(chromosome)
        anomaly_score = self._calculate_anomaly_detection_capability(chromosome)
        incident_score = self._calculate_incident_response_readiness(chromosome)

        # Calculate weighted overall score
        overall_score = (
            insider_score * self.weights['insider_threat'] +
            exposure_score * self.weights['data_exposure'] +
            escalation_score * self.weights['privilege_escalation'] +
            anomaly_score * self.weights['anomaly_detection'] +
            incident_score * self.weights['incident_response']
        )

        return RiskMetrics(
            insider_threat_score=insider_score,
            data_exposure_score=exposure_score,
            privilege_escalation_score=escalation_score,
            anomaly_detection_score=anomaly_score,
            incident_response_readiness=incident_score,
            overall_risk_score=overall_score,
            details={
                'high_risk_rule_count': self._count_high_risk_rules(chromosome),
                'mitigation_controls': self._count_mitigation_controls(chromosome),
                'risk_indicators': self._identify_risk_indicators(chromosome)
            }
        )

    def _calculate_insider_threat_risk(self, chromosome: PolicyChromosome) -> float:
        """Calculate insider threat risk level (0-1, higher = more risk)."""
        if not chromosome.rules:
            return 0.5

        risk_factors = []

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                # Check for overly broad access
                if self._is_broad_access(rule):
                    risk_factors.append(0.8)

                # Check for lack of time restrictions
                if not self._has_time_restriction(rule):
                    risk_factors.append(0.6)

                # Check for lack of location restrictions
                if not self._has_location_restriction(rule):
                    risk_factors.append(0.5)

                # Check for sensitive actions without approval
                if self._has_sensitive_actions(rule) and not self._requires_approval(rule):
                    risk_factors.append(0.9)

        if not risk_factors:
            return 0.1  # Low risk

        return sum(risk_factors) / len(risk_factors)

    def _is_broad_access(self, rule: PolicyRule) -> bool:
        """Check if rule grants overly broad access."""
        resource = rule.resource.value

        # Check for complete wildcards
        if resource == '*:*:*':
            return True

        # Check for wildcard actions
        if any(action.value == '*' for action in rule.actions):
            return True

        # Check for lack of conditions
        if len(rule.conditions) == 0:
            return True

        return False

    def _has_time_restriction(self, rule: PolicyRule) -> bool:
        """Check if rule has time-based restrictions."""
        time_fields = ['time', 'hour', 'day', 'date', 'schedule', 'expiry', 'valid_until']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(time_field in field for time_field in time_fields):
                    return True

        return False

    def _has_location_restriction(self, rule: PolicyRule) -> bool:
        """Check if rule has location-based restrictions."""
        location_fields = ['location', 'ip', 'country', 'region', 'office', 'vpn']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(loc_field in field for loc_field in location_fields):
                    return True

        return False

    def _has_sensitive_actions(self, rule: PolicyRule) -> bool:
        """Check if rule includes sensitive actions."""
        return any(
            action.value in self.sensitive_actions
            for action in rule.actions
        )

    def _requires_approval(self, rule: PolicyRule) -> bool:
        """Check if rule requires approval."""
        approval_fields = ['approval', 'approved', 'authorized', 'confirmed']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(approval in field for approval in approval_fields):
                    return True

        return False

    def _calculate_data_exposure_risk(self, chromosome: PolicyChromosome) -> float:
        """Calculate data exposure risk level (0-1, higher = more risk)."""
        if not chromosome.rules:
            return 0.5

        exposure_risks = []

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                resource = rule.resource.value

                # Check for PII/sensitive data access
                if self._is_sensitive_data(resource):
                    # No masking/encryption requirement
                    if not self._has_data_protection(rule):
                        exposure_risks.append(0.9)
                    else:
                        exposure_risks.append(0.3)

                # Check for export/download capabilities
                if self._allows_data_export(rule):
                    exposure_risks.append(0.7)

                # Check for bulk access
                if self._allows_bulk_access(rule):
                    exposure_risks.append(0.8)

        if not exposure_risks:
            return 0.1

        return sum(exposure_risks) / len(exposure_risks)

    def _is_sensitive_data(self, resource: str) -> bool:
        """Check if resource contains sensitive data."""
        sensitive_patterns = ['pii', 'personal', 'confidential', 'secret',
                            'private', 'sensitive', 'financial', 'medical']
        resource_lower = resource.lower()
        return any(pattern in resource_lower for pattern in sensitive_patterns)

    def _has_data_protection(self, rule: PolicyRule) -> bool:
        """Check if rule has data protection measures."""
        protection_fields = ['masked', 'encrypted', 'anonymized', 'redacted']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(protection in field for protection in protection_fields):
                    return True

        return False

    def _allows_data_export(self, rule: PolicyRule) -> bool:
        """Check if rule allows data export."""
        export_actions = ['export', 'download', 'transfer', 'copy']
        return any(
            action.value in export_actions
            for action in rule.actions
        )

    def _allows_bulk_access(self, rule: PolicyRule) -> bool:
        """Check if rule allows bulk data access."""
        # Wildcard in resource suggests bulk access
        if '*' in rule.resource.value:
            return True

        # Bulk actions
        bulk_actions = ['list_all', 'get_all', 'export_all', 'download_all']
        return any(
            action.value in bulk_actions
            for action in rule.actions
        )

    def _calculate_privilege_escalation_risk(self, chromosome: PolicyChromosome) -> float:
        """Calculate privilege escalation risk (0-1, higher = more risk)."""
        if not chromosome.rules:
            return 0.5

        escalation_risks = []

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                # Check for IAM/permission management access
                if self._is_permission_management(rule):
                    # Can grant to self?
                    if not self._prevents_self_grant(rule):
                        escalation_risks.append(0.9)
                    else:
                        escalation_risks.append(0.3)

                # Check for role assumption
                if self._allows_role_assumption(rule):
                    escalation_risks.append(0.7)

                # Check for security control modification
                if self._allows_security_modification(rule):
                    escalation_risks.append(0.8)

        if not escalation_risks:
            return 0.1

        return sum(escalation_risks) / len(escalation_risks)

    def _is_permission_management(self, rule: PolicyRule) -> bool:
        """Check if rule involves permission management."""
        perm_resources = ['iam', 'permissions', 'roles', 'policies', 'grants']
        resource_lower = rule.resource.value.lower()
        return any(perm in resource_lower for perm in perm_resources)

    def _prevents_self_grant(self, rule: PolicyRule) -> bool:
        """Check if rule prevents granting permissions to self."""
        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                operator = condition.value.get('operator')
                value = condition.value.get('value')

                if 'target' in field or 'grantee' in field:
                    if operator == ConditionOperator.NOT_EQUALS and value == 'self':
                        return True

        return False

    def _allows_role_assumption(self, rule: PolicyRule) -> bool:
        """Check if rule allows role assumption."""
        assume_actions = ['assume', 'switch', 'impersonate', 'become']
        return any(
            action.value in assume_actions
            for action in rule.actions
        )

    def _allows_security_modification(self, rule: PolicyRule) -> bool:
        """Check if rule allows modifying security controls."""
        security_resources = ['security', 'audit', 'logging', 'monitoring']
        modify_actions = ['modify', 'disable', 'delete', 'override']

        resource_match = any(
            sec in rule.resource.value.lower()
            for sec in security_resources
        )

        action_match = any(
            action.value in modify_actions
            for action in rule.actions
        )

        return resource_match and action_match

    def _calculate_anomaly_detection_capability(self, chromosome: PolicyChromosome) -> float:
        """Calculate anomaly detection capability score (0-1, higher = better)."""
        if not chromosome.rules:
            return 0.0

        detection_features = []

        # Check for behavioral conditions
        behavioral_score = self._score_behavioral_conditions(chromosome)
        detection_features.append(behavioral_score)

        # Check for rate limiting
        rate_limit_score = self._score_rate_limiting(chromosome)
        detection_features.append(rate_limit_score)

        # Check for unusual access patterns
        pattern_score = self._score_pattern_detection(chromosome)
        detection_features.append(pattern_score)

        return sum(detection_features) / len(detection_features)

    def _score_behavioral_conditions(self, chromosome: PolicyChromosome) -> float:
        """Score behavioral anomaly detection conditions."""
        behavioral_fields = ['normal_hours', 'usual_location', 'typical_behavior',
                           'risk_score', 'anomaly_score']

        behavioral_rules = 0
        for rule in chromosome.rules:
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    if any(beh in field for beh in behavioral_fields):
                        behavioral_rules += 1
                        break

        if not chromosome.rules:
            return 0.0

        return min(1.0, behavioral_rules / len(chromosome.rules))

    def _score_rate_limiting(self, chromosome: PolicyChromosome) -> float:
        """Score rate limiting implementation."""
        rate_fields = ['rate_limit', 'max_requests', 'throttle', 'quota']

        rate_limited_rules = 0
        for rule in chromosome.rules:
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    if any(rate in field for rate in rate_fields):
                        rate_limited_rules += 1
                        break

        if not chromosome.rules:
            return 0.0

        return min(1.0, rate_limited_rules / len(chromosome.rules))

    def _score_pattern_detection(self, chromosome: PolicyChromosome) -> float:
        """Score access pattern detection capabilities."""
        pattern_fields = ['access_pattern', 'unusual_activity', 'deviation',
                         'baseline', 'normal_pattern']

        pattern_aware_rules = 0
        for rule in chromosome.rules:
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    if any(pattern in field for pattern in pattern_fields):
                        pattern_aware_rules += 1
                        break

        if not chromosome.rules:
            return 0.0

        return min(1.0, pattern_aware_rules / len(chromosome.rules))

    def _calculate_incident_response_readiness(self, chromosome: PolicyChromosome) -> float:
        """Calculate incident response readiness score (0-1, higher = better)."""
        readiness_factors = []

        # Check for emergency access procedures
        emergency_score = self._score_emergency_access(chromosome)
        readiness_factors.append(emergency_score)

        # Check for audit trail quality
        audit_score = self._score_audit_trail_quality(chromosome)
        readiness_factors.append(audit_score)

        # Check for containment capabilities
        containment_score = self._score_containment_capabilities(chromosome)
        readiness_factors.append(containment_score)

        if not readiness_factors:
            return 0.0

        return sum(readiness_factors) / len(readiness_factors)

    def _score_emergency_access(self, chromosome: PolicyChromosome) -> float:
        """Score emergency access procedures."""
        emergency_rules = sum(
            1 for rule in chromosome.rules
            if 'emergency' in rule.resource.value.lower() or
            any('emergency' in str(c.value).lower() for c in rule.conditions)
        )

        if not chromosome.rules:
            return 0.0

        # Having some emergency procedures is good
        if emergency_rules > 0:
            return min(1.0, emergency_rules / 5)  # Cap at 5 emergency rules
        return 0.0

    def _score_audit_trail_quality(self, chromosome: PolicyChromosome) -> float:
        """Score audit trail quality for incident response."""
        audit_fields = ['audit', 'log', 'track', 'record', 'monitor']

        audit_rules = 0
        for rule in chromosome.rules:
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    if any(audit in field for audit in audit_fields):
                        audit_rules += 1
                        break

        if not chromosome.rules:
            return 0.0

        return min(1.0, audit_rules / len(chromosome.rules))

    def _score_containment_capabilities(self, chromosome: PolicyChromosome) -> float:
        """Score ability to contain incidents."""
        # Look for deny rules that can block access
        containment_rules = sum(
            1 for rule in chromosome.rules
            if rule.effect.value == PolicyEffect.DENY and
            any(action.value in ['*', 'all'] for action in rule.actions)
        )

        if not chromosome.rules:
            return 0.0

        # Having containment rules is good
        if containment_rules > 0:
            return min(1.0, containment_rules / 3)  # Cap at 3 containment rules
        return 0.0

    def _count_high_risk_rules(self, chromosome: PolicyChromosome) -> int:
        """Count number of high-risk rules."""
        high_risk_count = 0

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                # Check if it's a high-risk resource
                is_high_risk_resource = any(
                    self._matches_pattern(rule.resource.value, pattern)
                    for pattern in self.high_risk_resources
                )

                # Check if it has sensitive actions
                has_sensitive = self._has_sensitive_actions(rule)

                if is_high_risk_resource or has_sensitive:
                    high_risk_count += 1

        return high_risk_count

    def _count_mitigation_controls(self, chromosome: PolicyChromosome) -> int:
        """Count number of risk mitigation controls."""
        mitigation_count = 0

        for rule in chromosome.rules:
            # MFA requirements
            if any('mfa' in str(c.value).lower() for c in rule.conditions):
                mitigation_count += 1

            # Approval requirements
            if self._requires_approval(rule):
                mitigation_count += 1

            # Time restrictions
            if self._has_time_restriction(rule):
                mitigation_count += 1

            # Location restrictions
            if self._has_location_restriction(rule):
                mitigation_count += 1

        return mitigation_count

    def _identify_risk_indicators(self, chromosome: PolicyChromosome) -> List[str]:
        """Identify specific risk indicators in the policy."""
        indicators = []

        for rule in chromosome.rules:
            # Check for overly permissive rules
            if rule.resource.value == '*:*:*' and rule.effect.value == PolicyEffect.ALLOW:
                indicators.append(f"Overly permissive rule: {rule.rule_id}")

            # Check for missing conditions on sensitive resources
            if self._is_high_risk_resource(rule) and len(rule.conditions) == 0:
                indicators.append(f"High-risk resource without conditions: {rule.rule_id}")

            # Check for dangerous action combinations
            dangerous_combos = [
                ['read', 'delete'],
                ['create', 'approve'],
                ['modify', 'audit']
            ]
            actions = [a.value for a in rule.actions]
            for combo in dangerous_combos:
                if all(action in actions for action in combo):
                    indicators.append(f"Dangerous action combination in rule: {rule.rule_id}")

        return indicators

    def _is_high_risk_resource(self, rule: PolicyRule) -> bool:
        """Check if rule targets high-risk resources."""
        return any(
            self._matches_pattern(rule.resource.value, pattern)
            for pattern in self.high_risk_resources
        )

    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches a pattern with wildcards."""
        import re

        if '*' in pattern:
            regex_pattern = pattern.replace('*', '.*')
            regex_pattern = f"^{regex_pattern}$"
            return bool(re.match(regex_pattern, resource))
        return resource == pattern
