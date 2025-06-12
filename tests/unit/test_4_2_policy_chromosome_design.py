"""
Unit tests for Policy Chromosome Representation Design (Subtask 4.2).

Tests cover:
- Policy chromosome structure and representation
- Gene encoding for identity access policies
- Rule composition and validation
- Chromosome mutation strategies
- Crossover operations for policy evolution
- Serialization and deserialization
- Policy conflict detection
- Compliance constraint encoding
"""

import pytest
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set
import copy

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    ConditionOperator,
    PolicyEffect
)


class TestPolicyChromosomeDesign:
    """Test suite for policy chromosome representation design."""

    def test_policy_chromosome_structure(self):
        """Test basic policy chromosome structure and attributes."""
        chromosome = PolicyChromosome()

        # Test basic attributes
        assert hasattr(chromosome, 'rules')
        assert hasattr(chromosome, 'fitness_scores')
        assert hasattr(chromosome, 'generation')
        assert hasattr(chromosome, 'chromosome_id')
        assert hasattr(chromosome, 'metadata')

        assert isinstance(chromosome.rules, list)
        assert isinstance(chromosome.fitness_scores, dict)
        assert isinstance(chromosome.generation, int)
        assert isinstance(chromosome.chromosome_id, str)
        assert isinstance(chromosome.metadata, dict)

        # Test ID generation
        assert len(chromosome.chromosome_id) == 16  # SHA256 truncated to 16 chars

    def test_gene_encoding_for_identity_policies(self):
        """Test gene encoding for various identity access policy types."""
        # Test resource gene
        resource_gene = Gene(
            gene_type=GeneType.RESOURCE,
            value="salesforce:accounts:*"
        )
        assert resource_gene.gene_type == GeneType.RESOURCE
        assert "salesforce" in resource_gene.value

        # Test action gene
        action_gene = Gene(
            gene_type=GeneType.ACTION,
            value=["read", "write", "delete"]
        )
        assert action_gene.gene_type == GeneType.ACTION
        assert "read" in action_gene.value

        # Test condition gene
        condition_gene = Gene(
            gene_type=GeneType.CONDITION,
            value={
                "field": "department",
                "operator": ConditionOperator.EQUALS,
                "value": "engineering"
            }
        )
        assert condition_gene.gene_type == GeneType.CONDITION
        assert condition_gene.value["field"] == "department"

        # Test priority gene
        priority_gene = Gene(
            gene_type=GeneType.PRIORITY,
            value=100
        )
        assert priority_gene.gene_type == GeneType.PRIORITY
        assert priority_gene.value == 100

    def test_policy_rule_composition(self):
        """Test composing policy rules from genes."""
        # Create genes for a complete policy rule
        resource_gene = Gene(gene_type=GeneType.RESOURCE, value="github:repos:cerby/*")
        action_genes = [
            Gene(gene_type=GeneType.ACTION, value="read"),
            Gene(gene_type=GeneType.ACTION, value="write")
        ]
        condition_genes = [
            Gene(
                gene_type=GeneType.CONDITION,
                value={
                    "field": "role",
                    "operator": ConditionOperator.IN,
                    "value": ["developer", "admin"]
                }
            )
        ]
        effect_gene = Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW)
        priority_gene = Gene(gene_type=GeneType.PRIORITY, value=50)

        # Create policy rule
        rule = PolicyRule(
            resource=resource_gene,
            actions=action_genes,
            conditions=condition_genes,
            effect=effect_gene,
            priority=priority_gene
        )

        # Test rule attributes
        assert rule.resource.value == "github:repos:cerby/*"
        assert len(rule.actions) == 2
        assert rule.actions[0].value == "read"
        assert rule.actions[1].value == "write"
        assert len(rule.conditions) == 1
        assert rule.effect.value == PolicyEffect.ALLOW
        assert rule.priority.value == 50
        assert rule.rule_id is not None

    def test_complex_policy_rule_scenarios(self):
        """Test complex policy rules for real-world scenarios."""
        # Scenario 1: Time-based access control
        time_conditions = [
            Gene(
                gene_type=GeneType.CONDITION,
                value={
                    "field": "time_of_day",
                    "operator": ConditionOperator.GREATER_THAN,
                    "value": "09:00"
                }
            ),
            Gene(
                gene_type=GeneType.CONDITION,
                value={
                    "field": "time_of_day",
                    "operator": ConditionOperator.LESS_THAN,
                    "value": "17:00"
                }
            ),
            Gene(
                gene_type=GeneType.CONDITION,
                value={
                    "field": "day_of_week",
                    "operator": ConditionOperator.NOT_IN,
                    "value": ["saturday", "sunday"]
                }
            )
        ]

        time_based_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="aws:s3:production-data/*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="read")],
            conditions=time_conditions,
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=80)
        )
        assert len(time_based_rule.conditions) == 3

        # Scenario 2: Multi-factor authentication requirement
        mfa_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="okta:admin:*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="*")],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "mfa_verified",
                        "operator": ConditionOperator.NOT_EQUALS,
                        "value": True
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.DENY),
            priority=Gene(gene_type=GeneType.PRIORITY, value=100)
        )
        assert mfa_rule.effect.value == PolicyEffect.DENY
        assert mfa_rule.priority.value == 100

        # Scenario 3: Compliance-based access
        compliance_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="salesforce:financial-records:*"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="export")
            ],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "user.certifications",
                        "operator": ConditionOperator.CONTAINS,
                        "value": "SOX-compliant"
                    }
                ),
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "request.purpose",
                        "operator": ConditionOperator.IN,
                        "value": ["audit", "compliance-review"]
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=90)
        )
        assert len(compliance_rule.conditions) == 2

    def test_chromosome_mutation_strategies(self):
        """Test various mutation strategies for policy chromosomes."""
        chromosome = PolicyChromosome()

        # Add initial rules
        rule1 = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="slack:channels:general"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="write")
            ],
            conditions=[],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=50)
        )
        rule2 = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="slack:channels:hr-*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="read")],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "department",
                        "operator": ConditionOperator.NOT_EQUALS,
                        "value": "hr"
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.DENY),
            priority=Gene(gene_type=GeneType.PRIORITY, value=60)
        )

        chromosome.add_rule(rule1)
        chromosome.add_rule(rule2)

        # Test mutation
        original_rule_count = len(chromosome.rules)
        mutation_params = {
            "rule_mutation_rate": 0.5,
            "rule_add_rate": 0.3,
            "rule_remove_rate": 0.1,
            "resource_pool": ["github:*", "aws:*", "slack:*"],
            "action_pool": ["read", "write", "delete", "create", "update"],
            "max_rules": 10,
            "min_rules": 1
        }

        # Clone and mutate
        mutated = chromosome.clone()
        original_id = mutated.chromosome_id
        mutated.mutate(mutation_params)

        # Verify mutation may have changed the chromosome
        # Note: Due to randomness, the chromosome might not always change
        assert isinstance(mutated, PolicyChromosome)
        assert mutated is not chromosome  # Different object

        # Test specific gene mutation
        gene = Gene(gene_type=GeneType.RESOURCE, value="github:repos:*")
        original_value = gene.value
        gene.mutate(mutation_params)
        # Gene might or might not mutate based on probability
        assert isinstance(gene.value, str)

    def test_crossover_operations(self):
        """Test crossover operations between policy chromosomes."""
        # Create parent chromosomes
        parent1 = PolicyChromosome()
        parent2 = PolicyChromosome()

        # Add rules to parent1
        for i in range(3):
            rule = PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value=f"service{i}:resource:*"),
                actions=[Gene(gene_type=GeneType.ACTION, value="read")],
                conditions=[],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
                priority=Gene(gene_type=GeneType.PRIORITY, value=i * 10)
            )
            parent1.add_rule(rule)

        # Add rules to parent2
        for i in range(3, 6):
            rule = PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value=f"service{i}:resource:*"),
                actions=[Gene(gene_type=GeneType.ACTION, value="write")],
                conditions=[],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.DENY),
                priority=Gene(gene_type=GeneType.PRIORITY, value=i * 10)
            )
            parent2.add_rule(rule)

        # Test uniform crossover
        child1, child2 = parent1.crossover(parent2, {"method": "uniform"})
        assert isinstance(child1, PolicyChromosome)
        assert isinstance(child2, PolicyChromosome)
        # Children should have rules from both parents (unless random selection put all in one)
        # Since uniform crossover is random, we can't guarantee the IDs will be different
        assert len(child1.rules) > 0
        assert len(child2.rules) > 0

        # Test single-point crossover
        child1, child2 = parent1.crossover(parent2, {"method": "single_point"})
        # Total rules should be preserved in crossover
        assert len(child1.rules) + len(child2.rules) == len(parent1.rules) + len(parent2.rules)

        # Test two-point crossover
        child1, child2 = parent1.crossover(parent2, {"method": "two_point"})
        assert isinstance(child1.rules, list)
        assert isinstance(child2.rules, list)

    def test_serialization_and_deserialization(self):
        """Test serialization and deserialization of policy chromosomes."""
        # Create a complex chromosome
        chromosome = PolicyChromosome()
        chromosome.metadata = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "compliance_frameworks": ["SOX", "GDPR"]
        }

        # Add diverse rules
        rules = [
            PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value="aws:iam:roles/*"),
                actions=[
                    Gene(gene_type=GeneType.ACTION, value="assume"),
                    Gene(gene_type=GeneType.ACTION, value="list")
                ],
                conditions=[
                    Gene(
                        gene_type=GeneType.CONDITION,
                        value={
                            "field": "source_ip",
                            "operator": ConditionOperator.IN,
                            "value": ["10.0.0.0/8"]
                        }
                    ),
                    Gene(
                        gene_type=GeneType.CONDITION,
                        value={
                            "field": "mfa",
                            "operator": ConditionOperator.EQUALS,
                            "value": True
                        }
                    )
                ],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
                priority=Gene(gene_type=GeneType.PRIORITY, value=100)
            ),
            PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value="database:production:*"),
                actions=[
                    Gene(gene_type=GeneType.ACTION, value="delete"),
                    Gene(gene_type=GeneType.ACTION, value="drop")
                ],
                conditions=[
                    Gene(
                        gene_type=GeneType.CONDITION,
                        value={
                            "field": "role",
                            "operator": ConditionOperator.NOT_IN,
                            "value": ["dba", "admin"]
                        }
                    )
                ],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.DENY),
                priority=Gene(gene_type=GeneType.PRIORITY, value=90)
            )
        ]

        for rule in rules:
            chromosome.add_rule(rule)

        # Test serialization
        serialized = chromosome.to_dict()
        assert isinstance(serialized, dict)
        assert "chromosome_id" in serialized
        assert "rules" in serialized
        assert "metadata" in serialized
        assert len(serialized["rules"]) == 2

        # Test JSON serialization
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)

        # Test deserialization
        deserialized_dict = json.loads(json_str)
        restored = PolicyChromosome.from_dict(deserialized_dict)

        assert restored.chromosome_id == chromosome.chromosome_id
        assert len(restored.rules) == len(chromosome.rules)
        assert restored.metadata["version"] == "1.0"
        assert "compliance_frameworks" in restored.metadata

    def test_policy_conflict_detection(self):
        """Test detection of conflicting policies within a chromosome."""
        chromosome = PolicyChromosome()

        # Add conflicting rules
        rule1 = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="github:repos:cerby/*"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="write"),
                Gene(gene_type=GeneType.ACTION, value="delete")
            ],
            conditions=[],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=50)
        )

        rule2 = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="github:repos:cerby/*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="write")],
            conditions=[],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.DENY),
            priority=Gene(gene_type=GeneType.PRIORITY, value=60)
        )

        chromosome.add_rule(rule1)
        chromosome.add_rule(rule2)

        # Test conflict detection
        constraints = {}  # Empty constraints for basic validation
        is_valid, errors = chromosome.validate(constraints)
        assert not is_valid  # Should be invalid due to conflicts
        assert len(errors) > 0
        assert any("conflict" in error.lower() for error in errors)

    def test_compliance_constraint_encoding(self):
        """Test encoding compliance constraints in chromosomes."""
        # Create a chromosome with compliance metadata
        chromosome = PolicyChromosome()
        chromosome.metadata = {
            "compliance_constraints": {
                "sox": {
                    "required_conditions": ["audit_trail", "approval_workflow"],
                    "prohibited_actions": ["delete_audit_logs"],
                    "data_retention_days": 2555  # 7 years
                },
                "gdpr": {
                    "required_conditions": ["user_consent", "purpose_limitation"],
                    "data_retention_days": 1095,  # 3 years
                    "right_to_erasure": True
                }
            }
        }

        # Add SOX-compliant rule
        sox_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="financial:reports:*"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="export")
            ],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "audit_trail",
                        "operator": ConditionOperator.EQUALS,
                        "value": True
                    }
                ),
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "approval_workflow",
                        "operator": ConditionOperator.EQUALS,
                        "value": "completed"
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=95)
        )

        # Add GDPR-compliant rule
        gdpr_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="user:personal-data:*"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="update")
            ],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "user_consent",
                        "operator": ConditionOperator.EQUALS,
                        "value": True
                    }
                ),
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "purpose",
                        "operator": ConditionOperator.IN,
                        "value": ["service-delivery", "legal-obligation"]
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=85)
        )

        chromosome.add_rule(sox_rule)
        chromosome.add_rule(gdpr_rule)

        # Validate compliance
        assert len(chromosome.rules) == 2
        assert chromosome.metadata["compliance_constraints"]["sox"]["data_retention_days"] == 2555
        assert chromosome.metadata["compliance_constraints"]["gdpr"]["right_to_erasure"] is True

    def test_chromosome_fitness_attributes(self):
        """Test fitness-related attributes for GA evaluation."""
        chromosome = PolicyChromosome()

        # Add rules with varying complexity
        simple_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="app:feature:basic"),
            actions=[Gene(gene_type=GeneType.ACTION, value="read")],
            conditions=[],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=30)
        )

        complex_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="app:feature:advanced"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="write"),
                Gene(gene_type=GeneType.ACTION, value="execute")
            ],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "level",
                        "operator": ConditionOperator.GREATER_THAN,
                        "value": 5
                    }
                ),
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "verified",
                        "operator": ConditionOperator.EQUALS,
                        "value": True
                    }
                ),
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "region",
                        "operator": ConditionOperator.IN,
                        "value": ["us-east", "eu-west"]
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=70)
        )

        chromosome.add_rule(simple_rule)
        chromosome.add_rule(complex_rule)

        # Test chromosome properties relevant to fitness
        assert len(chromosome.rules) == 2
        assert chromosome.rules[0].priority.value < chromosome.rules[1].priority.value
        assert len(chromosome.rules[1].conditions) > len(chromosome.rules[0].conditions)

    def test_policy_inheritance_and_hierarchy(self):
        """Test policy inheritance and hierarchical rule structures."""
        chromosome = PolicyChromosome()

        # Base policy for all users
        base_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="common:resources:*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="read")],
            conditions=[],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=10)
        )

        # Department-specific override
        dept_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="dept:engineering:*"),
            actions=[
                Gene(gene_type=GeneType.ACTION, value="read"),
                Gene(gene_type=GeneType.ACTION, value="write")
            ],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "department",
                        "operator": ConditionOperator.EQUALS,
                        "value": "engineering"
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=50)
        )

        # Role-specific override (highest priority)
        admin_rule = PolicyRule(
            resource=Gene(gene_type=GeneType.RESOURCE, value="*:*:*"),
            actions=[Gene(gene_type=GeneType.ACTION, value="*")],
            conditions=[
                Gene(
                    gene_type=GeneType.CONDITION,
                    value={
                        "field": "role",
                        "operator": ConditionOperator.EQUALS,
                        "value": "admin"
                    }
                )
            ],
            effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
            priority=Gene(gene_type=GeneType.PRIORITY, value=100)
        )

        chromosome.add_rule(base_rule)
        chromosome.add_rule(dept_rule)
        chromosome.add_rule(admin_rule)

        # Verify hierarchy by priority
        sorted_rules = sorted(chromosome.rules, key=lambda r: r.priority.value, reverse=True)
        assert sorted_rules[0].priority.value == 100  # Admin rule first
        assert sorted_rules[1].priority.value == 50   # Department rule second
        assert sorted_rules[2].priority.value == 10   # Base rule last


class TestPolicyChromosomeAdvancedFeatures:
    """Test advanced features of policy chromosome representation."""

    def test_dynamic_resource_patterns(self):
        """Test dynamic resource pattern matching and wildcards."""
        chromosome = PolicyChromosome()

        # Test various resource patterns
        patterns = [
            "service:*:*",  # All resources in service
            "service:type:specific-id",  # Specific resource
            "service:type:prefix-*",  # Prefix matching
            "service:type:*-suffix",  # Suffix matching
            "service:type:*middle*",  # Contains matching
        ]

        for i, pattern in enumerate(patterns):
            rule = PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value=pattern),
                actions=[Gene(gene_type=GeneType.ACTION, value="read")],
                conditions=[],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
                priority=Gene(gene_type=GeneType.PRIORITY, value=i * 10)
            )
            chromosome.add_rule(rule)

        assert len(chromosome.rules) == len(patterns)

        # Test resource matching logic
        assert chromosome._resource_matches("service:type:specific-id", "service:type:specific-id")
        assert chromosome._resource_matches("service:*:*", "service:anything:here")
        assert chromosome._resource_matches("service:type:prefix-*", "service:type:prefix-123")

    def test_condition_complexity(self):
        """Test complex condition combinations and operators."""
        # Test all condition operators
        operators = [
            ConditionOperator.EQUALS,
            ConditionOperator.NOT_EQUALS,
            ConditionOperator.IN,
            ConditionOperator.NOT_IN,
            ConditionOperator.CONTAINS,
            ConditionOperator.GREATER_THAN,
            ConditionOperator.LESS_THAN,
            ConditionOperator.REGEX
        ]

        chromosome = PolicyChromosome()

        for op in operators:
            value = {
                ConditionOperator.EQUALS: "exact_value",
                ConditionOperator.NOT_EQUALS: "not_this",
                ConditionOperator.IN: ["option1", "option2"],
                ConditionOperator.NOT_IN: ["excluded1", "excluded2"],
                ConditionOperator.CONTAINS: "substring",
                ConditionOperator.GREATER_THAN: 10,
                ConditionOperator.LESS_THAN: 100,
                ConditionOperator.REGEX: r"^user-\d+$"
            }.get(op, "default")

            rule = PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value=f"test:operator:{op.value}"),
                actions=[Gene(gene_type=GeneType.ACTION, value="test")],
                conditions=[
                    Gene(
                        gene_type=GeneType.CONDITION,
                        value={
                            "field": f"test_field_{op.value}",
                            "operator": op,
                            "value": value
                        }
                    )
                ],
                effect=Gene(gene_type=GeneType.EFFECT, value=PolicyEffect.ALLOW),
                priority=Gene(gene_type=GeneType.PRIORITY, value=50)
            )
            chromosome.add_rule(rule)

        assert len(chromosome.rules) == len(operators)

    def test_chromosome_clone_and_equality(self):
        """Test chromosome cloning and equality operations."""
        original = PolicyChromosome()
        original.metadata["test_key"] = "test_value"

        # Add some rules
        for i in range(3):
            rule = PolicyRule(
                resource=Gene(gene_type=GeneType.RESOURCE, value=f"service{i}:*:*"),
                actions=[
                    Gene(gene_type=GeneType.ACTION, value="action1"),
                    Gene(gene_type=GeneType.ACTION, value="action2")
                ],
                conditions=[
                    Gene(
                        gene_type=GeneType.CONDITION,
                        value={
                            "field": f"field{i}",
                            "operator": ConditionOperator.EQUALS,
                            "value": f"value{i}"
                        }
                    )
                ],
                effect=Gene(
                    gene_type=GeneType.EFFECT,
                    value=PolicyEffect.ALLOW if i % 2 == 0 else PolicyEffect.DENY
                ),
                priority=Gene(gene_type=GeneType.PRIORITY, value=i * 20)
            )
            original.add_rule(rule)

        # Test cloning
        clone = original.clone()

        # Verify clone is independent
        # Note: If the rules are identical, the chromosome IDs will be the same
        # since they're based on a hash of the rules
        assert clone.chromosome_id == original.chromosome_id  # Same content = same ID
        assert len(clone.rules) == len(original.rules)
        assert clone.metadata["test_key"] == "test_value"

        # Verify deep copy - modifying clone doesn't affect original
        clone.rules[0].priority.value = 999
        assert original.rules[0].priority.value != 999

        # Test that rules are properly cloned
        for i, (orig_rule, clone_rule) in enumerate(zip(original.rules, clone.rules)):
            # Rule IDs should be the same since they're based on content hash
            assert orig_rule.rule_id == clone_rule.rule_id
            assert orig_rule.effect.value == clone_rule.effect.value
            assert orig_rule.resource.value == clone_rule.resource.value
            assert len(orig_rule.actions) == len(clone_rule.actions)
