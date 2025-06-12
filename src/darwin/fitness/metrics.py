"""
Fitness metrics module for Darwin genetic algorithm framework.

This module provides metric classes used by various fitness functions.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class SecurityMetrics:
    """Security-related metrics for policy evaluation."""
    privilege_score: float
    mfa_coverage: float
    sensitive_resource_protection: float
    condition_strength: float
    deny_rule_presence: float
    wildcard_usage: float
    overall_score: float
    details: Dict[str, Any]


@dataclass
class ProductivityMetrics:
    """Productivity-related metrics for policy evaluation."""
    access_coverage: float
    workflow_efficiency: float
    collaboration_score: float
    tool_availability: float
    rule_simplicity: float
    response_time_impact: float
    overall_score: float
    details: Dict[str, Any]


@dataclass
class ComplianceMetrics:
    """Compliance-related metrics for policy evaluation."""
    framework_scores: Dict[str, float]
    audit_trail_coverage: float
    data_protection_score: float
    access_control_maturity: float
    retention_compliance: float
    segregation_of_duties: float
    overall_score: float
    details: Dict[str, Any]


@dataclass
class RiskMetrics:
    """Risk assessment metrics for policy evaluation."""
    insider_threat_score: float
    data_exposure_score: float
    privilege_escalation_score: float
    anomaly_detection_score: float
    incident_response_readiness: float
    overall_risk_score: float
    details: Dict[str, Any]


@dataclass
class CoverageMetrics:
    """Policy coverage metrics."""
    resource_coverage: float
    action_coverage: float
    condition_coverage: float
    user_coverage: float
    overall_coverage: float
    uncovered_resources: List[str]
    details: Dict[str, Any]


@dataclass
class UXMetrics:
    """User experience metrics for policy evaluation."""
    clarity_score: float
    simplicity_score: float
    consistency_score: float
    discoverability_score: float
    error_prevention_score: float
    overall_ux_score: float
    details: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance impact metrics for policy evaluation."""
    evaluation_complexity: float
    caching_effectiveness: float
    rule_efficiency: float
    response_time_estimate: float
    scalability_score: float
    overall_performance: float
    details: Dict[str, Any]


@dataclass
class MultiObjectiveMetrics:
    """Combined metrics from multiple fitness objectives."""
    objective_scores: Dict[str, float]
    weighted_total: float
    pareto_rank: int
    dominance_count: int
    dominated_by_count: int
    details: Dict[str, Any]
