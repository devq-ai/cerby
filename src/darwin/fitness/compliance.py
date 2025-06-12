"""
Compliance fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate how well policies
adhere to regulatory compliance frameworks like SOX, GDPR, HIPAA, etc.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import re
from datetime import datetime

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    PolicyEffect,
    ConditionOperator
)
from src.darwin.fitness.base import FitnessFunction, FitnessMetrics


@dataclass
class ComplianceMetrics(FitnessMetrics):
    """Detailed compliance metrics for a policy chromosome."""
    framework_scores: Dict[str, float]
    audit_trail_coverage: float
    data_protection_score: float
    access_control_maturity: float
    retention_compliance: float
    segregation_of_duties: float


class ComplianceFitness(FitnessFunction):
    """
    Fitness function that evaluates compliance with regulatory frameworks.

    This function considers:
    - SOX compliance (audit trails, approvals, segregation of duties)
    - GDPR compliance (consent, data minimization, right to erasure)
    - HIPAA compliance (minimum necessary, encryption, access logging)
    - General compliance best practices
    """

    def __init__(self, frameworks: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize compliance fitness function.

        Args:
            frameworks: Dictionary mapping framework names to their configurations.
                       Each framework should have:
                       - weight: Relative importance of this framework
                       - requirements: List of required compliance features
                       - critical_resources: Resources that must be compliant
        """
        super().__init__()

        if frameworks is None:
            # Default frameworks with equal weight
            frameworks = {
                "sox": {
                    "weight": 0.33,
                    "requirements": ["audit_trail", "approval_workflow", "segregation_of_duties"],
                    "critical_resources": ["financial:*", "accounting:*", "audit:*"]
                },
                "gdpr": {
                    "weight": 0.33,
                    "requirements": ["user_consent", "data_minimization", "right_to_erasure"],
                    "critical_resources": ["user:personal-data:*", "customer:pii:*"]
                },
                "hipaa": {
                    "weight": 0.34,
                    "requirements": ["encryption", "access_logging", "minimum_necessary"],
                    "critical_resources": ["patient:*", "medical:*", "health:*"]
                }
            }

        self.frameworks = frameworks

        # Normalize weights
        total_weight = sum(f.get("weight", 1.0) for f in frameworks.values())
        for framework in frameworks.values():
            framework["weight"] = framework.get("weight", 1.0) / total_weight

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the compliance fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Compliance fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> ComplianceMetrics:
        """
        Calculate detailed compliance metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed compliance metrics
        """
        # Calculate framework-specific scores
        framework_scores = {}
        weighted_total = 0.0

        for framework_name, framework_config in self.frameworks.items():
            score = self._evaluate_framework_compliance(chromosome, framework_name, framework_config)
            framework_scores[framework_name] = score
            weighted_total += score * framework_config["weight"]

        # Calculate general compliance metrics
        audit_trail = self._calculate_audit_trail_coverage(chromosome)
        data_protection = self._calculate_data_protection_score(chromosome)
        access_maturity = self._calculate_access_control_maturity(chromosome)
        retention = self._calculate_retention_compliance(chromosome)
        segregation = self._calculate_segregation_of_duties(chromosome)

        # Combine framework scores with general compliance
        general_score = (
            audit_trail * 0.25 +
            data_protection * 0.25 +
            access_maturity * 0.20 +
            retention * 0.15 +
            segregation * 0.15
        )

        # Final score: 70% framework-specific, 30% general compliance
        overall_score = weighted_total * 0.7 + general_score * 0.3

        return ComplianceMetrics(
            score=overall_score,
            details={
                'framework_scores': framework_scores,
                'audit_trail_coverage': audit_trail,
                'data_protection_score': data_protection,
                'access_control_maturity': access_maturity,
                'retention_compliance': retention,
                'segregation_of_duties': segregation
            },
            framework_scores=framework_scores,
            audit_trail_coverage=audit_trail,
            data_protection_score=data_protection,
            access_control_maturity=access_maturity,
            retention_compliance=retention,
            segregation_of_duties=segregation
        )

    def _evaluate_framework_compliance(self, chromosome: PolicyChromosome,
                                     framework_name: str,
                                     framework_config: Dict[str, Any]) -> float:
        """Evaluate compliance with a specific framework."""
        requirements = framework_config.get("requirements", [])
        critical_resources = framework_config.get("critical_resources", [])

        if not requirements:
            return 1.0

        requirement_scores = []

        # Check each requirement
        for requirement in requirements:
            if requirement == "audit_trail":
                score = self._check_audit_trail_requirement(chromosome, critical_resources)
            elif requirement == "approval_workflow":
                score = self._check_approval_workflow(chromosome, critical_resources)
            elif requirement == "segregation_of_duties":
                score = self._check_segregation_of_duties(chromosome, critical_resources)
            elif requirement == "user_consent":
                score = self._check_user_consent(chromosome, critical_resources)
            elif requirement == "data_minimization":
                score = self._check_data_minimization(chromosome, critical_resources)
            elif requirement == "right_to_erasure":
                score = self._check_right_to_erasure(chromosome, critical_resources)
            elif requirement == "encryption":
                score = self._check_encryption_requirement(chromosome, critical_resources)
            elif requirement == "access_logging":
                score = self._check_access_logging(chromosome, critical_resources)
            elif requirement == "minimum_necessary":
                score = self._check_minimum_necessary(chromosome, critical_resources)
            else:
                score = 0.5  # Unknown requirement

            requirement_scores.append(score)

        return sum(requirement_scores) / len(requirement_scores)

    def _check_audit_trail_requirement(self, chromosome: PolicyChromosome,
                                     critical_resources: List[str]) -> float:
        """Check if audit trail requirements are met."""
        if not critical_resources:
            return 1.0

        covered = 0
        total = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._matches_critical_resource(resource, critical_resources):
                total += 1
                if self._has_audit_condition(rule):
                    covered += 1

        if total == 0:
            return 1.0  # No critical resources found

        return covered / total

    def _has_audit_condition(self, rule: PolicyRule) -> bool:
        """Check if rule has audit-related conditions."""
        audit_fields = ['audit_trail', 'audit', 'logging_enabled', 'track_access']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(audit in field for audit in audit_fields):
                    value = condition.value.get('value')
                    if value is True or value == 'enabled':
                        return True

        return False

    def _check_approval_workflow(self, chromosome: PolicyChromosome,
                               critical_resources: List[str]) -> float:
        """Check if approval workflows are required for critical resources."""
        if not critical_resources:
            return 1.0

        approval_score = 0.0
        rule_count = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._matches_critical_resource(resource, critical_resources):
                rule_count += 1

                # Check for approval conditions
                has_approval = any(
                    self._is_approval_condition(c) for c in rule.conditions
                )

                # High-risk actions should require approval
                high_risk_actions = ['delete', 'modify', 'approve', 'transfer']
                has_high_risk = any(
                    action.value in high_risk_actions for action in rule.actions
                )

                if has_high_risk and has_approval:
                    approval_score += 1.0
                elif has_high_risk and not has_approval:
                    approval_score += 0.0  # Penalty for missing approval
                elif not has_high_risk and has_approval:
                    approval_score += 0.8  # Good but not required
                else:
                    approval_score += 0.6  # Low risk, no approval needed

        if rule_count == 0:
            return 1.0

        return approval_score / rule_count

    def _is_approval_condition(self, condition: Gene) -> bool:
        """Check if a condition relates to approval."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            approval_fields = ['approval', 'approved', 'authorized', 'sign_off']
            return any(field_name in field for field_name in approval_fields)
        return False

    def _check_segregation_of_duties(self, chromosome: PolicyChromosome,
                                   critical_resources: List[str]) -> float:
        """Check segregation of duties implementation."""
        segregation_score = 0.0
        relevant_rules = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._matches_critical_resource(resource, critical_resources):
                relevant_rules += 1

                # Check for self-restriction conditions
                has_self_restriction = any(
                    self._is_self_restriction(c) for c in rule.conditions
                )

                # Actions that should have segregation
                segregated_actions = ['approve', 'authorize', 'sign', 'validate']
                needs_segregation = any(
                    action.value in segregated_actions for action in rule.actions
                )

                if needs_segregation and has_self_restriction:
                    segregation_score += 1.0
                elif needs_segregation and not has_self_restriction:
                    segregation_score += 0.2  # Poor segregation
                else:
                    segregation_score += 0.8  # Action doesn't require segregation

        if relevant_rules == 0:
            return 1.0

        return segregation_score / relevant_rules

    def _is_self_restriction(self, condition: Gene) -> bool:
        """Check if condition prevents self-actions."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            operator = condition.value.get('operator')
            value = condition.value.get('value')

            if ('self' in field or 'requester' in field or 'initiator' in field):
                if operator == ConditionOperator.NOT_EQUALS and value == 'self':
                    return True

        return False

    def _check_user_consent(self, chromosome: PolicyChromosome,
                          critical_resources: List[str]) -> float:
        """Check GDPR user consent requirements."""
        consent_score = 0.0
        personal_data_rules = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._is_personal_data_resource(resource) or \
               self._matches_critical_resource(resource, critical_resources):
                personal_data_rules += 1

                # Check for consent conditions
                has_consent = any(
                    self._is_consent_condition(c) for c in rule.conditions
                )

                if has_consent:
                    consent_score += 1.0
                else:
                    # Check if it's a necessary operation without consent
                    necessary_actions = ['delete', 'anonymize']
                    if any(action.value in necessary_actions for action in rule.actions):
                        consent_score += 0.8  # Acceptable for data deletion
                    else:
                        consent_score += 0.2  # Missing consent

        if personal_data_rules == 0:
            return 1.0

        return consent_score / personal_data_rules

    def _is_personal_data_resource(self, resource: str) -> bool:
        """Check if resource contains personal data."""
        personal_indicators = ['personal', 'pii', 'user-data', 'customer-data',
                              'profile', 'identity', 'private']
        resource_lower = resource.lower()
        return any(indicator in resource_lower for indicator in personal_indicators)

    def _is_consent_condition(self, condition: Gene) -> bool:
        """Check if condition relates to user consent."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            value = condition.value.get('value')

            consent_fields = ['consent', 'agreed', 'permission', 'authorized_by_user']
            if any(consent in field for consent in consent_fields):
                return value is True or value == 'granted'

        return False

    def _check_data_minimization(self, chromosome: PolicyChromosome,
                               critical_resources: List[str]) -> float:
        """Check GDPR data minimization principle."""
        minimization_score = 0.0
        data_access_rules = 0

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                resource = rule.resource.value
                if self._is_data_resource(resource):
                    data_access_rules += 1

                    # Check if access is limited
                    is_limited = self._is_limited_access(rule)

                    # Check if there's a purpose limitation
                    has_purpose = any(
                        self._is_purpose_condition(c) for c in rule.conditions
                    )

                    if is_limited and has_purpose:
                        minimization_score += 1.0
                    elif is_limited or has_purpose:
                        minimization_score += 0.7
                    else:
                        minimization_score += 0.3

        if data_access_rules == 0:
            return 1.0

        return minimization_score / data_access_rules

    def _is_data_resource(self, resource: str) -> bool:
        """Check if resource is a data resource."""
        data_indicators = ['data', 'records', 'information', 'documents', 'files']
        resource_lower = resource.lower()
        return any(indicator in resource_lower for indicator in data_indicators)

    def _is_limited_access(self, rule: PolicyRule) -> bool:
        """Check if rule provides limited access."""
        # Specific resources (not wildcards) indicate limited access
        resource = rule.resource.value
        if '*' not in resource:
            return True

        # Limited actions indicate controlled access
        limited_actions = ['read', 'view', 'list']
        all_limited = all(
            action.value in limited_actions for action in rule.actions
        )

        return all_limited

    def _is_purpose_condition(self, condition: Gene) -> bool:
        """Check if condition specifies purpose."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            purpose_fields = ['purpose', 'reason', 'justification', 'use_case']
            return any(purpose in field for purpose in purpose_fields)
        return False

    def _check_right_to_erasure(self, chromosome: PolicyChromosome,
                              critical_resources: List[str]) -> float:
        """Check GDPR right to erasure implementation."""
        erasure_capability = 0.0
        personal_data_types = 0

        # Find all personal data resource types
        personal_resources = set()
        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._is_personal_data_resource(resource):
                # Extract resource type
                parts = resource.split(':')
                if parts:
                    personal_resources.add(parts[0])

        personal_data_types = len(personal_resources)

        if personal_data_types == 0:
            return 1.0

        # Check if deletion is allowed for each type
        for resource_type in personal_resources:
            has_delete = False
            for rule in chromosome.rules:
                if rule.effect.value == PolicyEffect.ALLOW:
                    resource = rule.resource.value
                    if resource_type in resource:
                        if any(action.value in ['delete', 'erase', 'remove']
                              for action in rule.actions):
                            has_delete = True
                            break

            if has_delete:
                erasure_capability += 1.0

        return erasure_capability / personal_data_types

    def _check_encryption_requirement(self, chromosome: PolicyChromosome,
                                    critical_resources: List[str]) -> float:
        """Check HIPAA encryption requirements."""
        encryption_score = 0.0
        sensitive_rules = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._is_health_resource(resource) or \
               self._matches_critical_resource(resource, critical_resources):
                sensitive_rules += 1

                # Check for encryption conditions
                has_encryption = any(
                    self._is_encryption_condition(c) for c in rule.conditions
                )

                # Write operations must have encryption
                write_actions = ['write', 'create', 'update', 'store']
                has_write = any(
                    action.value in write_actions for action in rule.actions
                )

                if has_write and has_encryption:
                    encryption_score += 1.0
                elif not has_write and has_encryption:
                    encryption_score += 0.9  # Good practice
                elif has_write and not has_encryption:
                    encryption_score += 0.2  # Missing required encryption
                else:
                    encryption_score += 0.7  # Read without encryption is acceptable

        if sensitive_rules == 0:
            return 1.0

        return encryption_score / sensitive_rules

    def _is_health_resource(self, resource: str) -> bool:
        """Check if resource contains health information."""
        health_indicators = ['health', 'medical', 'patient', 'clinical',
                           'diagnosis', 'treatment', 'prescription']
        resource_lower = resource.lower()
        return any(indicator in resource_lower for indicator in health_indicators)

    def _is_encryption_condition(self, condition: Gene) -> bool:
        """Check if condition requires encryption."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            value = condition.value.get('value')

            encryption_fields = ['encrypted', 'encryption', 'secure_channel', 'tls']
            if any(enc in field for enc in encryption_fields):
                return value is True or value == 'enabled'

        return False

    def _check_access_logging(self, chromosome: PolicyChromosome,
                            critical_resources: List[str]) -> float:
        """Check HIPAA access logging requirements."""
        return self._check_audit_trail_requirement(chromosome, critical_resources)

    def _check_minimum_necessary(self, chromosome: PolicyChromosome,
                               critical_resources: List[str]) -> float:
        """Check HIPAA minimum necessary standard."""
        # Similar to data minimization but specific to healthcare
        return self._check_data_minimization(chromosome, critical_resources)

    def _matches_critical_resource(self, resource: str,
                                 critical_patterns: List[str]) -> bool:
        """Check if resource matches any critical resource pattern."""
        for pattern in critical_patterns:
            if self._matches_pattern(resource, pattern):
                return True
        return False

    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches a pattern with wildcards."""
        if '*' in pattern:
            regex_pattern = pattern.replace('*', '.*')
            regex_pattern = f"^{regex_pattern}$"
            return bool(re.match(regex_pattern, resource))
        return resource == pattern

    def _calculate_audit_trail_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate overall audit trail coverage."""
        if not chromosome.rules:
            return 0.0

        rules_with_audit = sum(
            1 for rule in chromosome.rules if self._has_audit_condition(rule)
        )

        return rules_with_audit / len(chromosome.rules)

    def _calculate_data_protection_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate overall data protection score."""
        protection_aspects = []

        # Check encryption usage
        encryption_rules = sum(
            1 for rule in chromosome.rules
            if any(self._is_encryption_condition(c) for c in rule.conditions)
        )
        if chromosome.rules:
            protection_aspects.append(encryption_rules / len(chromosome.rules))

        # Check access restrictions
        restricted_rules = sum(
            1 for rule in chromosome.rules
            if len(rule.conditions) >= 2  # Multiple conditions indicate restrictions
        )
        if chromosome.rules:
            protection_aspects.append(restricted_rules / len(chromosome.rules))

        # Check data classification awareness
        classified_rules = sum(
            1 for rule in chromosome.rules
            if any(self._is_classification_condition(c) for c in rule.conditions)
        )
        if chromosome.rules:
            protection_aspects.append(classified_rules / len(chromosome.rules))

        if not protection_aspects:
            return 0.0

        return sum(protection_aspects) / len(protection_aspects)

    def _is_classification_condition(self, condition: Gene) -> bool:
        """Check if condition relates to data classification."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            classification_fields = ['classification', 'sensitivity', 'confidentiality',
                                   'data_level', 'security_level']
            return any(cls in field for cls in classification_fields)
        return False

    def _calculate_access_control_maturity(self, chromosome: PolicyChromosome) -> float:
        """Calculate access control maturity level."""
        if not chromosome.rules:
            return 0.0

        maturity_scores = []

        # Granular permissions (not wildcards)
        granular_rules = sum(
            1 for rule in chromosome.rules
            if '*' not in rule.resource.value
        )
        maturity_scores.append(granular_rules / len(chromosome.rules))

        # Condition-based access
        conditional_rules = sum(
            1 for rule in chromosome.rules
            if len(rule.conditions) > 0
        )
        maturity_scores.append(conditional_rules / len(chromosome.rules))

        # Deny rules present (defense in depth)
        deny_rules = sum(
            1 for rule in chromosome.rules
            if rule.effect.value == PolicyEffect.DENY
        )
        ideal_deny_ratio = 0.2  # 20% deny rules is ideal
        actual_ratio = deny_rules / len(chromosome.rules)
        deny_score = 1.0 - abs(actual_ratio - ideal_deny_ratio) / ideal_deny_ratio
        maturity_scores.append(max(0, deny_score))

        # Priority differentiation
        priorities = [rule.priority.value for rule in chromosome.rules]
        if len(set(priorities)) > 1:  # Multiple priority levels
            maturity_scores.append(1.0)
        else:
            maturity_scores.append(0.5)

        return sum(maturity_scores) / len(maturity_scores)

    def _calculate_retention_compliance(self, chromosome: PolicyChromosome) -> float:
        """Calculate data retention compliance score."""
        retention_aware_rules = 0
        data_rules = 0

        for rule in chromosome.rules:
            if self._is_data_resource(rule.resource.value):
                data_rules += 1

                # Check for retention-related conditions
                has_retention = any(
                    self._is_retention_condition(c) for c in rule.conditions
                )

                if has_retention:
                    retention_aware_rules += 1

        if data_rules == 0:
            return 1.0

        # Base score from retention awareness
        base_score = retention_aware_rules / data_rules

        # Bonus for having deletion capabilities
        has_deletion = any(
            rule.effect.value == PolicyEffect.ALLOW and
            any(action.value in ['delete', 'purge', 'expire'] for action in rule.actions)
            for rule in chromosome.rules
        )

        if has_deletion:
            base_score = min(1.0, base_score * 1.2)

        return base_score

    def _is_retention_condition(self, condition: Gene) -> bool:
        """Check if condition relates to data retention."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            retention_fields = ['retention', 'expiry', 'expire', 'age',
                              'created_before', 'retention_period']
            return any(ret in field for ret in retention_fields)
        return False

    def _calculate_segregation_of_duties(self, chromosome: PolicyChromosome) -> float:
        """Calculate overall segregation of duties score."""
        segregation_score = 0.0
        relevant_rules = 0

        # Actions that typically require segregation
        segregated_actions = ['approve', 'authorize', 'sign', 'validate',
                            'transfer', 'pay', 'release']

        for rule in chromosome.rules:
            if any(action.value in segregated_actions for action in rule.actions):
                relevant_rules += 1

                # Check for self-restriction
                has_self_restriction = any(
                    self._is_self_restriction(c) for c in rule.conditions
                )

                # Check for dual control
                has_dual_control = any(
                    self._is_dual_control_condition(c) for c in rule.conditions
                )

                if has_self_restriction and has_dual_control:
                    segregation_score += 1.0
                elif has_self_restriction or has_dual_control:
                    segregation_score += 0.7
                else:
                    segregation_score += 0.2

        if relevant_rules == 0:
            return 1.0

        return segregation_score / relevant_rules

    def _is_dual_control_condition(self, condition: Gene) -> bool:
        """Check if condition requires dual control."""
        if isinstance(condition.value, dict):
            field = condition.value.get('field', '').lower()
            dual_fields = ['dual_control', 'two_person', 'second_approval',
                         'co_signer', 'witness']
            return any(dual in field for dual in dual_fields)
        return False
