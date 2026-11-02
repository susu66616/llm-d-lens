import json

import pytest
import yaml

from llm_d_lens.configuration.guide_settings import validate_guide_settings


def manifest(*, cpu_bytes=2147483648, nic_count=2):
    return yaml.safe_dump_all(
        [
            {
                "kind": "Deployment",
                "metadata": {"name": "decode"},
                "spec": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "modelserver",
                                    "command": ["vllm", "serve"],
                                    "args": [
                                        "New/Model",
                                        "--block-size=32",
                                        "--kv-transfer-config="
                                        + json.dumps(
                                            {
                                                "kv_connector": "OffloadingConnector",
                                                "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
                                            }
                                        ),
                                    ],
                                }
                            ]
                        }
                    },
                },
            },
            {
                "kind": "ResourceClaimTemplate",
                "spec": {
                    "spec": {
                        "devices": {"requests": [{"exactly": {"deviceClassName": "dranet-rdma", "count": nic_count}}]}
                    }
                },
            },
        ]
    )


def test_cpu_capacity_must_match_native_connector():
    content = {"guideVariant": "native/cpu/base", "guideSettings": {"cacheCpuGiB": 2}}
    validate_guide_settings(content, manifest(), "tiered-prefix-cache")
    with pytest.raises(ValueError, match="CPU cache"):
        validate_guide_settings(content, manifest(cpu_bytes=1), "tiered-prefix-cache")


def test_explicit_nic_capacity_must_match_device_requests():
    content = {"hardware_profile": "intel-xpu", "guideVariant": "vllm-rdma", "guideSettings": {"rdmaNicCount": 2}}
    validate_guide_settings(content, manifest(), "pd-disaggregation")
    with pytest.raises(ValueError, match="NIC"):
        validate_guide_settings(content, manifest(nic_count=1), "pd-disaggregation")


def test_router_values_require_a_saved_deployment_bundle():
    with pytest.raises(ValueError, match="bundle"):
        validate_guide_settings({"guideSettings": {"routerValues": "router: {}"}}, manifest(), "optimized-baseline")


@pytest.mark.parametrize("settings", [{"unexpected": True}, {"cacheCpuGiB": -1}, {"rdmaNicCount": True}])
def test_unknown_or_invalid_guide_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_guide_settings({"guideSettings": settings}, manifest(), "optimized-baseline")


def test_typed_router_settings_target_cpu_producer_and_reject_yaml_conflicts():
    from llm_d_lens.configuration.guide_settings import apply_router_settings

    plugins = {
        "plugins": [
            {"type": "approx-prefix-cache-producer", "name": "gpu-prefix-cache-producer"},
            {"type": "approx-prefix-cache-producer", "name": "cpu-prefix-cache-producer"},
        ]
    }
    values = {
        "router": {
            "epp": {
                "pluginsConfigFile": "plugins.yaml",
                "pluginsCustomConfig": {"plugins.yaml": yaml.safe_dump(plugins)},
            }
        }
    }
    apply_router_settings(values, {"cacheLruCapacity": 123})
    rendered = yaml.safe_load(values["router"]["epp"]["pluginsCustomConfig"]["plugins.yaml"])["plugins"]
    assert "parameters" not in rendered[0]
    assert rendered[1]["parameters"] == {"lruCapacityPerServer": 123}
    with pytest.raises(ValueError, match="conflict"):
        apply_router_settings(values, {"cacheLruCapacity": 10, "routerValues": yaml.safe_dump(values)})


def test_cpu_index_capacity_requires_saved_router_inputs():
    with pytest.raises(ValueError, match="bundle"):
        validate_guide_settings({"guideSettings": {"cacheLruCapacity": 123}}, manifest(), "tiered-prefix-cache")
