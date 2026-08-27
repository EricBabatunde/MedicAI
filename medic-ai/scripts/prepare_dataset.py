# ============================================================
# BEFORE RUNNING THIS SCRIPT, INSTALL DEPENDENCIES:
#
#   pip install datasets pandas
#
# Then execute:
#
#   python3 scripts/prepare_dataset.py
# ============================================================

import json
import os

from datasets import load_dataset


def main():
    # ------------------------------------------------------------------
    # 1. Download / load the MedQA-USMLE 4-option dataset (train split)
    # ------------------------------------------------------------------
    print("[INFO] Downloading MedQA-USMLE-4-options dataset (train split)...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    print(f"[INFO] Dataset loaded. Total records available: {len(dataset)}")

    # Output path – write into the project's data/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, "medqa_rag_formatted.jsonl")

    max_records = 5000
    success_count = 0

    # ------------------------------------------------------------------
    # 2. Process records and write JSONL
    # ------------------------------------------------------------------
    print(f"[INFO] Processing up to {max_records} records...")

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, record in enumerate(dataset):
            if idx >= max_records:
                break

            # --- Extract fields ---
            question = record["question"]
            options = record["options"]
            answer_idx = record["answer_idx"]
            answer = record["answer"]

            # --- Simulated RAG context ---
            options_text = "\n".join(
                f"  {key}. {value}" for key, value in options.items()
            )
            simulated_context = (
                f"Clinical vignette: {question}\n\n"
                f"Differential options:\n{options_text}\n\n"
                f"Correct clinical determination: {answer} (Option {answer_idx})"
            )

            # --- User message ---
            user_content = f"Context: {simulated_context}\n\nQuery: {question}"

            # --- Assistant structured JSON response ---
            assistant_json = {
                "clinical_assessment": (
                    f"Based on the clinical presentation, the most likely "
                    f"diagnosis/answer is: {answer}."
                ),
                "treatment_plan": (
                    f"The recommended course of action aligns with: {answer}. "
                    f"Further workup and management should follow evidence-based "
                    f"guidelines for this condition."
                ),
                "critical_warnings": (
                    "Always correlate with the full clinical picture. "
                    "This response is generated for educational purposes and "
                    "must not replace professional medical judgement."
                ),
                "references": [
                    "MedQA-USMLE dataset (GBaker/MedQA-USMLE-4-options)",
                    "USMLE Step 1 / Step 2 CK review material",
                ],
            }

            # --- ChatML payload ---
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical RAG assistant. Respond with "
                            "strict JSON only. Do not wrap your response in "
                            "markdown code fences or add any text outside the "
                            "JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_json),
                    },
                ]
            }

            f.write(json.dumps(payload) + "\n")
            success_count += 1

            # Progress indicator every 500 records
            if (idx + 1) % 500 == 0:
                print(f"  [PROGRESS] Processed {idx + 1} records...")

    print(f"[SUCCESS] Wrote {success_count} records to {output_file}")


if __name__ == "__main__":
    main()

