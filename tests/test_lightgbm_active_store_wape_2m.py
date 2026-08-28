from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "21_train_lightgbm_v2_active_store_2m_wape.py"


def load_script():
    spec = spec_from_file_location("active_store_wape_2m", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_2m_candidate_search_covers_internal_and_upper_range():
    module = load_script()
    candidates = module.candidate_iterations()
    assert candidates == sorted(set(candidates))
    assert candidates[0] == 100
    assert 700 in candidates
    assert 850 in candidates
    assert candidates[-1] == 2000
    assert len(candidates) == 11


def test_2m_outputs_are_isolated_from_1m_and_old_models():
    module = load_script()
    paths = module.output_paths()
    assert all("2m" in path.name.lower() for path in paths.values())
    assert all("active_store" in path.name.lower() for path in paths.values())
    assert paths["final_model"] != ROOT / "models/final/lightgbm_v2_logl2_2m.txt"


def test_2m_uses_complete_horizon_eligibility():
    module = load_script()
    assert module.HORIZON == "2m"
    assert module.EXPECTED_VALID_ROWS == 18_491_375
    assert module.EXPECTED_TEST_ROWS == 12_859_664
