# cline_utils/dependency_system/analysis/local_llm_processor.py

import logging
import re
from typing import Any, Optional, Tuple
import torch
import gc
from cline_utils.dependency_system.utils.resource_validator import ResourceValidator

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)


class LocalLLMProcessor:
    """
    Handles local LLM interactions to determine dependencies between files.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048):
        super().__init__()
        self.model_path = model_path
        self.max_n_ctx = 32768
        self.current_n_ctx = n_ctx
        self._current_n_gpu_layers: int = 0
        self._model: Optional[Any] = None
        self._pinned_state: Optional[Any] = None

    def save_pinned_state(self):
        """Saves the current KV cache state (e.g., after processing a system prompt)."""
        if self._model and hasattr(self._model, "save_state"):
            self._pinned_state = self._model.save_state()
            logger.info("KV cache state pinned successfully.")

    def restore_pinned_state(self):
        """Restores the previously pinned KV cache state."""
        if self._model and self._pinned_state and hasattr(self._model, "load_state"):
            self._model.load_state(self._pinned_state)
            logger.info("Pinned KV cache state restored.")

    def _load_model(self, required_ctx: int):
        # Context sizing strategy: dynamic "orbiting" allocation
        # The context window orbits the actual needed size at a fixed radius,
        # both increasing AND decreasing as requirements change.

        # Buffer margin to avoid edge cases (5% or minimum 128 tokens)
        ctx_buffer = max(128, int(required_ctx * 0.05))
        # Round up to nearest 512 for memory alignment efficiency
        round_to = 512
        # Threshold for context reduction (don't shrink unless savings exceed 15%)
        shrink_threshold = 0.15

        # Calculate precise context with buffer
        needed = required_ctx + ctx_buffer
        # Round up to nearest multiple of round_to
        n_ctx = ((needed + round_to - 1) // round_to) * round_to

        # Clamp to maximum allowed context
        n_ctx = min(n_ctx, self.max_n_ctx)

        if self._model is not None:
            current = self.current_n_ctx
            # Calculate if we should reload (either direction)
            if current >= n_ctx:
                # Current context is sufficient - check if we should shrink
                # Only shrink if the difference is significant (exceeds threshold)
                excess_ratio = (current - n_ctx) / current if current > 0 else 0
                if excess_ratio <= shrink_threshold:
                    # Not enough savings to justify reload, keep current
                    return self._model
                # Significant savings - reload with smaller context
                logger.info(
                    f"Reloading model to reduce n_ctx from {current} to {n_ctx} "
                    f"(saving {excess_ratio*100:.1f}% context)"
                )
            else:
                # Need more context - must reload
                logger.info(
                    f"Reloading model to increase n_ctx from {current} to {n_ctx}"
                )

            self.close()

        if Llama is None:
            raise ImportError("llama-cpp-python is not installed.")

        # --- Dynamic GPU/CPU Splitting ---
        n_gpu_layers = -1
        try:
            validator = ResourceValidator()
            gpu_stats = validator.validate_gpu()
            if gpu_stats.get("gpu_available"):
                vram_available_mb = gpu_stats.get("vram_available_mb", 0.0)
                if vram_available_mb > 0:
                    MB_PER_LAYER = 125

                    # Empirical estimation parameters
                    base_overhead_mb = 30
                    mb_per_1k_tokens = 120

                    safety_buffer_mb = max(
                        500, vram_available_mb * 0.1
                    )  # Minimum 500MB safety buffer

                    # Calculate estimated memory for context
                    context_memory_mb = (
                        base_overhead_mb + (n_ctx / 1000.0) * mb_per_1k_tokens
                    )

                    ideal_vram_mb = (
                        context_memory_mb + (36 * MB_PER_LAYER) + safety_buffer_mb
                    )

                    if vram_available_mb >= ideal_vram_mb:
                        n_gpu_layers = 36
                        vram_ratio = 1.0
                    else:
                        # Gradient allocation ensures any free VRAM contributes to GPU layers
                        # rather than strictly zeroing out if buffer thresholds aren't met
                        vram_ratio = (
                            vram_available_mb / ideal_vram_mb
                            if ideal_vram_mb > 0
                            else 1.0
                        )
                        n_gpu_layers = max(0, min(36, int(36 * vram_ratio)))

                    logger.info(
                        f"Dynamic VRAM check: {vram_available_mb}MB free. "
                        f"Est. Context: {context_memory_mb:.1f}MB, Reserve: {safety_buffer_mb:.1f}MB. "
                        f"Ratio: {vram_ratio:.2f}. "
                        f"Assigning {n_gpu_layers} layers to GPU."
                    )
        except Exception as e:
            logger.warning(
                f"Failed to dynamically calculate VRAM layer split: {e}. Defaulting to full GPU offload."
            )

        logger.info(
            f"Loading local LLM from {self.model_path} with n_ctx={n_ctx} and n_gpu_layers={n_gpu_layers}..."
        )
        self._model = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            type_k=8,
            type_v=8,
            flash_attn=True,
        )
        self.current_n_ctx = n_ctx
        self._current_n_gpu_layers = n_gpu_layers
        return self._model

    def get_token_count(self, text: str) -> int:
        """Get exact token count using the model's tokenizer."""
        model = self._load_model(2048)  # Ensure at least base context loaded
        try:
            tokens = model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Tokenizer failed: {e}. Falling back to estimate.")
            return len(text) // 4

    def determine_dependency(
        self,
        source_content: str,
        target_content: str,
        source_basename: str,
        target_basename: str,
        source_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
        instructional_prompt: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Determines the dependency character between two files.

        Returns:
            Tuple[str, str]: A tuple containing (dependency_char, full_response_text)
        """
        if instructional_prompt is None:
            instructional_prompt = """
*   **Determine Relationship (CRITICAL STEP)**: Based on file contents, determine the **true relational necessity or essential conceptual link** between the source (`key_string`) and each target key being verified.
    *   **Go Beyond Semantic Similarity**: Suggestions might only indicate related topics. However, if either file defines the "Why," the rules, or the architecture for the other, it is an essential dependency.
*   **Focus on Relational and Contextual Necessity**:
    *   **Logic & Purpose**: Does the *row file* provide the business logic, requirements, or purpose that the *column file* implements? (Leads to 'd' or '<').
    *   **Technical Reliance**: Does the code in the *row file* directly **import, call, or inherit from** code in the *column file*? (Leads to '<' or 'x').
    *   **Knowledge Requirement**: Does a developer/LLM need to read the *column file* to safely or correctly modify the *row file*? (Leads to '<' or 'd').
    *   **Implementation Link**: Is the *row file* **essential documentation** for understanding or implementing the concepts/code in the *column file*? (Leads to 'd' or '>').
    *   **Architectural Fit**: Are these files part of the same specific feature or architectural pattern where changing one without the other would cause conceptual drift or technical debt? (Leads to 'x', '<', '>', or 'd').
*   **Purpose of Dependencies**: Verified dependencies guide the **Strategy phase** and the **Execution phase**. A dependency should mean "This file is part of the necessary context required to work effectively on the other."
*   **Assign 'n' ONLY for Unrelated Content**: If the relationship is purely coincidental, uses similar common terms in a different context, or is an unrelated file, assign 'n'. **If there is any doubt regarding conceptual relevance, err on the side of 'd' (Documentation/Conceptual link) rather than 'n'.**

**Dependency Criteria**
    * '<' (Row Requires Column): Row relies on Column for context, logic, or operation
    * '>' (Column Requires Row): Column relies on Row for context, logic, or operation
    * 'x' (Mutual Requirement): Mutual reliance or deep conceptual link requiring co-consideration
    * 'd' (Documentation/Conceptual): Row is documentation or defines the "Why/How" essential for Column (or vice-versa)
    * 'n' (Verified No Dependency): Confirmed NO relational, functional, or conceptual link exists

**Instructions**
1. Analyze the relationship (Functional, Logical, Conceptual, Architectural) between source and target
2. Determine the dependency character that best represents the relationship (<, >, x, d, n)
3. State your reasoning for the chosen dependency type, keep it concise and to the point
4. Provide a summary in the format:
   Dependency Verification Results:

   [Source File] [Target File] -> [Dependency Character]
   Reasoning: [Your reasoning]

Important Notes:
- Focus on Relational Necessity: Does one file provide the context, blueprint, or understanding for the other?
- Files that share a logical flow or implementation goal indicate a positive dependency.
- If a clear directional dependency exists 'x', '<', or '>' should be prioritized.
- DO NOT add decorators or other characters (``, **, etc) to the Dependency Character.

Expected Output:
Clear summary of dependency determination in the format dictated in instruction 4.
            """

        # Base overhead for prompt structure
        # Hardcoded based on measurement of the static template (810) + response margin
        wrapper_tokens = 810
        response_margin = 520

        # Initial estimate to decide n_ctx
        # If tokens provided, use them, otherwise use char count / 4
        s_est = source_tokens or (len(source_content) // 4)
        t_est = target_tokens or (len(target_content) // 4)

        initial_required = s_est + t_est + wrapper_tokens + response_margin
        model = self._load_model(initial_required)

        # Now get EXACT counts and truncate if needed to fit in current model's window
        # We use a loop to ensure we fit, as truncation is character-based
        max_attempts = 3
        final_prompt = ""
        for attempt in range(max_attempts):
            current_prompt = (
                f"<|im_start|>system\n{instructional_prompt}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Source File: {source_basename}\n"
                f"```\n{source_content}\n```\n\n"
                f"Target File: {target_basename}\n"
                f"```\n{target_content}\n```\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Optimization: If it's the first attempt and we have tokens, skip re-tokenization
            if attempt == 0 and source_tokens is not None and target_tokens is not None:
                prompt_tokens = wrapper_tokens + source_tokens + target_tokens
            else:
                prompt_tokens = self.get_token_count(current_prompt)

            # Add margin for response
            effective_ctx = self.current_n_ctx - response_margin

            if prompt_tokens <= effective_ctx:
                final_prompt = current_prompt
                break

            if attempt == max_attempts - 1:
                logger.error(
                    f"Failed to truncate prompt to fit context. Prompt: {prompt_tokens}, Ctx: {self.current_n_ctx}"
                )
                raise ValueError(
                    f"Prompt tokens ({prompt_tokens}) exceed context window of {self.current_n_ctx} even after truncation."
                )

            # Truncate
            logger.warning(
                f"Prompt tokens ({prompt_tokens}) exceed context window ({self.current_n_ctx}). Truncating attempt {attempt+1}."
            )

            # Allocation proportionality
            available_for_files = int((effective_ctx - wrapper_tokens) * 0.95)
            s_len = len(source_content)
            t_len = len(target_content)
            total_len = s_len + t_len

            s_max_chars = int(available_for_files * 4 * (s_len / total_len))
            t_max_chars = int(available_for_files * 4 * (t_len / total_len))

            source_content = source_content[:s_max_chars] + "... [TRUNCATED]"
            target_content = target_content[:t_max_chars] + "... [TRUNCATED]"

        output = model(final_prompt, max_tokens=500, stop=["<|im_end|>"], echo=False)

        result_text = output["choices"][0]["text"].strip()

        # Parse for the expected output format:
        # The line above "Reasoning:" always ends with the dependency character
        valid_chars = ["<", ">", "x", "d", "n"]

        lines = result_text.split("\n")
        reasoning_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("Reasoning:"):
                reasoning_idx = i
                break

        if reasoning_idx > 0:
            for j in range(reasoning_idx - 1, -1, -1):
                prev_line = lines[j].strip()
                if prev_line:
                    # Look for -> followed by our character, allowing quotes/backticks
                    match = re.search(r"->\s*[`'\"*]*([<>xdn])", prev_line)
                    if match:
                        return match.group(1), result_text

                    # Fallback: just check the end of the cleaned line
                    clean_line = prev_line.strip("`'\"* ")
                    if clean_line and clean_line[-1] in valid_chars:
                        return clean_line[-1], result_text
                    break  # Stop at the first non-empty line above "Reasoning:"

        safe_text = result_text.encode("ascii", "backslashreplace").decode("ascii")
        logger.warning(f"Unexpected model output: {safe_text}. Defaulting to 'p'.")
        return "p", result_text

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._current_n_gpu_layers = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
