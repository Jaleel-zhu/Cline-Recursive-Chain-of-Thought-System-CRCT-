# core/analysis_state_manager.py
"""
Analysis State Manager for tracking execution milestones of analyze-project.
Enables idempotent resumption and prevents generational key map desynchronization
caused by premature process terminations (e.g., timeouts during heavy phases).
"""

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

STATE_FILENAME = "analysis_state.json"


class AnalysisPhase(str, Enum):
    """Phases of project analysis execution."""

    INIT = "init"
    KEYS_GENERATED = "keys_generated"
    FILES_ANALYZED = "files_analyzed"
    SYMBOLS_MERGED = "symbols_merged"
    EMBEDDINGS_GENERATED = "embeddings_generated"
    SUGGESTIONS_COMPLETED = "suggestions_completed"
    TRACKERS_COMMITTED = "trackers_committed"


class AnalysisStateManager:
    """Manages persistent analysis state and recovery across runs."""

    def __init__(self, core_dir: Optional[str] = None):
        self.core_dir = os.path.abspath(core_dir) if core_dir else None

        from cline_utils.dependency_system.core import resolve_state_path

        self.state_file_path = normalize_path(
            resolve_state_path(STATE_FILENAME, self.core_dir)
        )
        self.current_state: Optional[Dict[str, Any]] = None

    def get_state_path(self) -> str:
        """Return the normalized path to the state file."""
        return self.state_file_path

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from disk if it exists."""
        if not os.path.exists(self.state_file_path):
            return None
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.current_state = data
                    return data
        except Exception as e:
            logger.warning(f"Failed to read analysis state from {self.state_file_path}: {e}")
        return None

    def start_or_resume_run(
        self, project_root: str, force: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an interrupted run can be resumed, or initialize a fresh run.

        Returns:
            Tuple of (is_resuming, state_dict)
        """
        norm_root = normalize_path(os.path.abspath(project_root))
        existing_state = self.load_state()

        if (
            not force
            and existing_state
            and existing_state.get("status") == "in_progress"
            and existing_state.get("project_root") == norm_root
        ):
            logger.info(
                f"Detected interrupted analysis run '{existing_state.get('run_id')}' "
                f"at phase '{existing_state.get('phase')}'. Resuming."
            )
            return True, existing_state

        if existing_state and existing_state.get("status") == "in_progress":
            logger.info(
                f"Found stale interrupted run '{existing_state.get('run_id')}'. "
                f"Rollback safety active; initializing clean run."
            )
            self.rollback_interrupted_run()

        # Initialize fresh state
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        new_state: Dict[str, Any] = {
            "run_id": run_id,
            "status": "in_progress",
            "phase": AnalysisPhase.INIT.value,
            "project_root": norm_root,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        self._save_state(new_state)
        self.current_state = new_state
        return False, new_state

    def record_phase(
        self, phase: AnalysisPhase, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record transition to a new analysis phase."""
        if not self.current_state:
            self.current_state = self.load_state() or {}

        phase_value = phase.value if isinstance(phase, AnalysisPhase) else str(phase)
        self.current_state["phase"] = phase_value
        self.current_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        if metadata:
            meta = self.current_state.setdefault("metadata", {})
            meta.update(metadata)

        try:
            self._save_state(self.current_state)
            logger.debug(f"Analysis checkpoint recorded: {phase.value}")
        except Exception as e:
            logger.warning(
                f"Could not persist analysis checkpoint '{phase.value}': {e}"
            )

    def rollback_interrupted_run(self) -> bool:
        """
        If an interrupted run is discarded, restore global_key_map.json from
        global_key_map_old.json so that the baseline generation matching
        existing trackers is preserved.
        """
        from cline_utils.dependency_system.core import resolve_state_path
        from cline_utils.dependency_system.core.key_manager import (
            GLOBAL_KEY_MAP_FILENAME,
            OLD_GLOBAL_KEY_MAP_FILENAME,
        )

        current_map_path = normalize_path(
            resolve_state_path(GLOBAL_KEY_MAP_FILENAME, self.core_dir)
        )
        old_map_path = normalize_path(
            resolve_state_path(OLD_GLOBAL_KEY_MAP_FILENAME, self.core_dir)
        )

        if os.path.exists(old_map_path):
            try:
                shutil.copyfile(old_map_path, current_map_path)
                logger.info(
                    f"Rollback restored '{GLOBAL_KEY_MAP_FILENAME}' from '{OLD_GLOBAL_KEY_MAP_FILENAME}'."
                )
                return True
            except OSError as e:
                logger.error(f"Failed to rollback key map: {e}")
        return False

    def mark_completed(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark the active analysis run as completed and committed."""
        if not self.current_state:
            self.current_state = self.load_state() or {}

        self.current_state["status"] = "completed"
        self.current_state["phase"] = AnalysisPhase.TRACKERS_COMMITTED.value
        self.current_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.current_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        if metadata:
            meta = self.current_state.setdefault("metadata", {})
            meta.update(metadata)

        self._save_state(self.current_state)
        logger.info(
            f"Analysis run '{self.current_state.get('run_id')}' successfully marked completed."
        )

    def clear_state(self) -> None:
        """Remove state file."""
        if os.path.exists(self.state_file_path):
            try:
                os.remove(self.state_file_path)
                self.current_state = None
            except OSError as e:
                logger.warning(f"Could not remove state file {self.state_file_path}: {e}")

    def _save_state(
        self, state_dict: Dict[str, Any], max_retries: int = 5, base_delay: float = 0.05
    ) -> None:
        """Atomically persist state dictionary to disk with retry on transient locks."""
        target_dir = os.path.dirname(self.state_file_path)
        os.makedirs(target_dir, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="state_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)

            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, self.state_file_path)
                    return
                except OSError as err:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2**attempt))
                    else:
                        raise err
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.warning(f"Failed to write state file {self.state_file_path}: {e}")
            raise
