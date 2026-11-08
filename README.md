# Post-training

A minimal interactive matrix of typical and alternative offline and online recipes for SFT, rejection sampling fine-tuning (RFT), DPO, hard and soft distillation, PPO, GRPO, and CISPO.

Live site: [Post-training](https://slinkoigor.github.io/llm-post-training-map/).

Open [the standalone HTML](dist/post-training-map.html) in a browser. It works offline, with its scripts, styles, and math fonts embedded.

## Interaction

- Each quadrant is a single vertical list. All thirteen cards retain their positions.
- Every card names its Offline or Online mode. Unoutlined rows show typical recipes; dashed outlines mark alternative recipes.
- Click a method name or its header to reveal the formula; click again to close it.
- Expand all reveals every formula with the method name above it.
- Hover or click a component to highlight the same component across the map and its rendered equations. Escape clears a selection.
- Dots indicate critic requirements. The information button opens notation, placement assumptions, and primary sources.
- Every pipeline ends with a loss → Actor LLM update. The stronger blue Actor LLM block is the trainable policy πθ; click it to highlight the actor in every card and formula. For distillation, this is the student model. PPO, GRPO, and CISPO cards show πθ explicitly in the importance-ratio definition.
- Blue blocks identify LLM-based components: Actor LLM, Teacher LLM, Ref LLM, Critic LLM, and Reward LLM. The reward-source switch changes the evaluator in RFT, PPO, GRPO, and CISPO between Reward LLM and a non-neural Verifier.
- Objectives are shown without line breaks inside expressions. Independent definitions appear on separate lines.

Cards are created once. Expanding, collapsing, and highlighting update existing DOM nodes without replacing, moving, hiding, or dimming other methods.

## Scope

The horizontal axis distinguishes fixed training data (Offline) from training data refreshed as the model changes (Online). Offline training may use multiple epochs. Online/offline describe the data-collection loop; on-policy/off-policy describe the relationship between the sampling and trainable policies.

The map uses conventional recipes as defaults: offline SFT, RFT, DPO, and teacher-data distillation; online PPO, GRPO, and CISPO. Dashed cards show supported alternatives: online SFT with fresh labels, online DPO with fresh preference pairs, iterative RFT with fresh reward-filtered rollouts, and hard/soft distillation on student-generated prefixes. These labels describe the shown recipes rather than immutable properties or measured prevalence of whole algorithm families.

The vertical axis distinguishes recipes that use explicit reward scores from those that use supplied training targets. Scores can come from a correctness Verifier or a learned Reward LLM, as selected by the reward-source switch. RFT uses scores to filter responses; PPO, GRPO, and CISPO use them to update the policy. DPO uses supplied preference pairs, SFT uses supplied target tokens, and distillation uses teacher outputs without explicit reward scores.

The online hard-distillation card uses teacher token labels on student-generated prefixes; online soft distillation uses direct KL matching (GKD). Both detach the sampling step. The offline distillation cards use fixed teacher-generated responses or prefixes. Policy-gradient distillation is a different variant.

PPO, GRPO, and CISPO use recent rollout snapshots and importance corrections. Their Online placement describes repeated rollout collection, not equality between the sampling and trainable policies at every update.

Hard distillation shares token cross-entropy with SFT and RFT; soft distillation matches teacher distributions with KL. The online SFT, RFT, and DPO variants keep the fitting objective while refreshing the dataset D_k between rounds. PPO, GRPO, and CISPO use the same explicit ρ^clip notation, with clipping bounds defined in each card. PPO and GRPO show min(ρA, ρ^clip A) directly in their objectives; CISPO uses sg(ρ^clip) as the weight on A log πθ. GRPO and CISPO share group-relative advantages. CISPO omits the reference KL penalty in the original recipe.

## Development

GitHub Pages publishes the repository root from the `main` branch. Push changes to `main` to update the live site.

Serve the source directory:

    python3 -m http.server 8054 --bind 127.0.0.1

Rebuild the portable HTML:

    python3 build.py

KaTeX 0.16.22 is vendored under its MIT license.
