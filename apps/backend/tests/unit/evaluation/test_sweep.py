"""Search domains, resource constraints and budget semantics."""

import pytest

from llm_d_lens.evaluation.sweep import generate_candidates, select_candidate
from llm_d_lens.evaluation.sweep_models import SweepDefinition, SweepDimension

BASE = {"guide": "optimized-baseline", "modelServer": "vllm", "replicas": 1, "tensorParallelSize": 1}


def definition(**overrides):
    return SweepDefinition.model_validate(
        {
            "dimensions": {
                "tensorParallelSize": {"minimum": 1, "maximum": 8},
                "replicas": {"minimum": 1, "maximum": 8},
            },
            "accelerator_budget": 8,
            **overrides,
        }
    )


def test_fixed_budget_filters_combinations_without_power_of_two_assumption():
    spec = definition(strategy="grid", resource_mode="fixed")
    plan = generate_candidates(spec, BASE)
    assert {(c["parameters"]["tensorParallelSize"], c["parameters"]["replicas"]) for c in plan["candidates"]} == {
        (1, 8),
        (2, 4),
        (4, 2),
        (8, 1),
    }
    assert plan["excluded"]
    six = generate_candidates(definition(strategy="grid", resource_mode="fixed", accelerator_budget=6), BASE)
    assert any(c["parameters"]["tensorParallelSize"] == 3 for c in six["candidates"])


def test_grid_never_silently_truncates_to_budget():
    with pytest.raises(ValueError, match="Complete grid"):
        generate_candidates(definition(strategy="grid", max_trials=1), BASE)


def test_ranges_reject_fractional_integer_values_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="integer range"):
        SweepDimension(minimum=1.5, maximum=4).points(integer=True, grid=True)
    with pytest.raises(ValueError):
        SweepDimension(minimum=float("nan"), maximum=4)


def test_pd_counts_both_roles():
    spec = definition(
        resource_mode="fixed", accelerator_budget=4, dimensions={"prefillReplicas": {"minimum": 1, "maximum": 4}}
    )
    plan = generate_candidates(
        spec, {**BASE, "guide": "pd-disaggregation", "replicas": 2, "prefillTensorParallelSize": 2}
    )
    with pytest.raises(ValueError, match="No candidates"):
        generate_candidates(spec, {**BASE, "guide": "pd-disaggregation", "prefillTensorParallelSize": 4})
    assert all(c["accelerator_count"] == 4 for c in plan["candidates"])


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        generate_candidates(definition(dimensions={"arbitrary": {"minimum": 1, "maximum": 2}}), BASE)


def test_selection_is_reproducible_and_stops_at_budget():
    spec = definition(max_trials=2)
    candidates = generate_candidates(spec, BASE)["candidates"]
    first = select_candidate(spec, candidates)
    assert first == select_candidate(spec, candidates)
    first.update(status="succeeded", score=10)
    second = select_candidate(spec, candidates)
    assert second != first
    second.update(status="failed")
    assert select_candidate(spec, candidates) is None


def test_automatic_search_reserves_confirmation_inside_trial_budget():
    spec = definition(max_trials=3, confirmation_trials=1)
    candidates = generate_candidates(spec, BASE)["candidates"]
    first = select_candidate(spec, candidates)
    first.update(status="succeeded", score=10)
    second = select_candidate(spec, candidates)
    second.update(status="succeeded", score=5)
    confirmation = select_candidate(spec, candidates)
    assert confirmation["confirmation_of"] == first["id"]
    assert confirmation["parameters"] == first["parameters"]
    confirmation.update(status="succeeded", score=9)
    assert select_candidate(spec, candidates) is None


def test_advanced_categorical_dimensions_do_not_require_numeric_values():
    spec = definition(dimensions={"argument:both:enable-prefix-caching": {"values": [True, False]}})
    candidates = generate_candidates(spec, BASE)["candidates"]
    assert len(candidates) == 2
    assert select_candidate(spec, candidates) is not None


def test_specified_candidates_cannot_truncate_fractional_topology():
    spec = definition(
        strategy="specified",
        specified_candidates=[{"replicas": 1.5}],
        dimensions={"replicas": {"minimum": 1, "maximum": 3}},
    )
    with pytest.raises(ValueError, match="integer"):
        generate_candidates(spec, BASE)


def test_pd_slices_apply_to_the_entire_topology_budget():
    spec = definition(strategy="grid", dimensions={"guide:slices": {"values": [1, 2, 3]}}, accelerator_budget=4)
    plan = generate_candidates(
        spec, {**BASE, "guide": "pd-disaggregation", "prefillReplicas": 1, "prefillTensorParallelSize": 1}
    )
    assert [item["accelerator_count"] for item in plan["candidates"]] == [2, 4]
    assert len(plan["excluded"]) == 1
    with pytest.raises(ValueError, match="PD"):
        generate_candidates(spec, BASE)
