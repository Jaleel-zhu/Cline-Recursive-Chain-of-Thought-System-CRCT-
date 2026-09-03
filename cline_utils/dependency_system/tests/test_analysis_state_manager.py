# tests/test_analysis_state_manager.py
import json
import os
from pathlib import Path

import pytest

from cline_utils.dependency_system.core.analysis_state_manager import (
    AnalysisPhase,
    AnalysisStateManager,
)


def test_state_manager_fresh_run(tmp_path: Path) -> None:
    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    is_resuming, state = state_mgr.start_or_resume_run(str(tmp_path))

    assert not is_resuming
    assert state["status"] == "in_progress"
    assert state["phase"] == AnalysisPhase.INIT.value
    assert state["project_root"] == str(tmp_path).replace("\\", "/")
    assert os.path.exists(state_mgr.get_state_path())


def test_state_manager_record_phase(tmp_path: Path) -> None:
    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    state_mgr.start_or_resume_run(str(tmp_path))

    state_mgr.record_phase(AnalysisPhase.KEYS_GENERATED, {"key_count": 42})
    state = state_mgr.load_state()
    assert state is not None
    assert state["phase"] == AnalysisPhase.KEYS_GENERATED.value
    assert state["metadata"]["key_count"] == 42

    state_mgr.record_phase(AnalysisPhase.EMBEDDINGS_GENERATED)
    state = state_mgr.load_state()
    assert state is not None
    assert state["phase"] == AnalysisPhase.EMBEDDINGS_GENERATED.value
    assert state["metadata"]["key_count"] == 42


def test_state_manager_resume_interrupted_run(tmp_path: Path) -> None:
    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    _, orig_state = state_mgr.start_or_resume_run(str(tmp_path))
    state_mgr.record_phase(AnalysisPhase.EMBEDDINGS_GENERATED)

    # Next invocation with same project root should detect resume
    new_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    is_resuming, resumed_state = new_mgr.start_or_resume_run(str(tmp_path))

    assert is_resuming
    assert resumed_state["run_id"] == orig_state["run_id"]
    assert resumed_state["phase"] == AnalysisPhase.EMBEDDINGS_GENERATED.value


def test_state_manager_force_does_not_resume(tmp_path: Path) -> None:
    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    _, orig_state = state_mgr.start_or_resume_run(str(tmp_path))
    state_mgr.record_phase(AnalysisPhase.EMBEDDINGS_GENERATED)

    # Force should start a new run
    new_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    is_resuming, new_state = new_mgr.start_or_resume_run(str(tmp_path), force=True)

    assert not is_resuming
    assert new_state["run_id"] != orig_state["run_id"]
    assert new_state["phase"] == AnalysisPhase.INIT.value


def test_state_manager_mark_completed(tmp_path: Path) -> None:
    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    state_mgr.start_or_resume_run(str(tmp_path))
    state_mgr.mark_completed({"trackers_updated": 3})

    state = state_mgr.load_state()
    assert state is not None
    assert state["status"] == "completed"
    assert state["phase"] == AnalysisPhase.TRACKERS_COMMITTED.value
    assert state["metadata"]["trackers_updated"] == 3

    # Subsequent run starts fresh because previous run was completed
    is_resuming, fresh_state = state_mgr.start_or_resume_run(str(tmp_path))
    assert not is_resuming


def test_state_manager_rollback_on_abandoned_run(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    current_map = state_dir / "global_key_map.json"
    old_map = state_dir / "global_key_map_old.json"

    # Simulate G1 in current and G0 in old
    old_map.write_text(json.dumps({"1A1": {"path": "baseline.py"}}), encoding="utf-8")
    current_map.write_text(json.dumps({"1A1": {"path": "shifted.py"}}), encoding="utf-8")

    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    state_mgr.start_or_resume_run(str(tmp_path))
    state_mgr.record_phase(AnalysisPhase.KEYS_GENERATED)

    # Now force a clean start; should rollback current_map to old_map (G0)
    state_mgr.start_or_resume_run(str(tmp_path), force=True)

    restored = json.loads(current_map.read_text(encoding="utf-8"))
    assert restored["1A1"]["path"] == "baseline.py"
