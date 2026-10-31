# LLM-as-a-judge: incomplete, follow-up planned

**When:** 29 October 2026. **Status:** incomplete. **Data:** [llm-judge.json](../results/llm-judge.json)

## Why a judge

Retrieval metrics show whether the right evidence was found, not whether the answer built on it is complete, grounded and properly cited. Grading hundreds of answers by hand is slow. A calibrated LLM judge can grade them on fixed dimensions (completeness, grounding, citation locality, calibration, language, source qualification, dialogue context) and give an overall "fully accepted" label.

## What happened

An earlier hosted configuration generated answers for 159 questions in each of four retrieval setups (arms): 549 unique answers covering 636 arm-question evaluations. Before use, the `gpt-6-sol` judge agreed with reference labels on 28 of 30 acceptance decisions.

Grading stopped when the provider enforced the project spend limit:

- 305 valid grades
- 242 rejected by the quota (HTTP 429)
- 2 internally inconsistent grades, counted as invalid

Quota rejections are missing grades, not failed answers.

## Why no winner is claimed

Valid coverage per arm ranged from 23 to 157. Grading stopped nonrandomly, so comparing acceptance rates across such unequal samples would be unfair. Only 10 questions were graded for every arm. On those, 7 or 8 of 10 answers were fully accepted in each arm. Per-case outcomes are in the JSON. These 10 are all answerable, exposed development or diagnostic cases. They cannot pick a winner across 159 questions.

The judge's grades are provisional model judgments, not human gold. The judge model also generated the answers, and the reference labels were agent adjudications.

## Follow-up

The 244 missing judgments (242 quota rejections plus 2 inconsistent) will be graded with the same judge model, prompt and schema, so that all 636 evaluations share one judge. The complete results will then be published, with either a justified winner or a plain statement that none emerged. Switching judges is not an option: a different judge would have to be calibrated and then regrade all 549 answers.
