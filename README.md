# Post-training

A minimal interactive matrix of SFT, rejection sampling fine-tuning (RFT), DPO, hard and soft distillation on external and student-generated prefixes, PPO, GRPO, and CISPO.

Live site: [Post-training](https://slinkoigor.github.io/llm-post-training-map/).

Open [the standalone HTML](dist/post-training-map.html) in a browser. It works offline, with its scripts, styles, and math fonts embedded.

## Interaction

- Each quadrant is a single vertical list. All ten cards retain their positions.
- Click a method name or its header to reveal the formula; click again to close it.
- Expand all reveals every formula with the method name above it.
- Hover or click a component to highlight the same component across the map and its rendered equations. Escape clears a selection.
- Dots indicate critic requirements. The information button opens notation, placement assumptions, and primary sources.
- Every pipeline ends with a loss → Actor LLM update. The stronger blue Actor LLM block is the trainable policy πθ; click it to highlight the actor in every card and formula. For distillation, this is the student model. RL cards show πθ explicitly in the importance-ratio definition.
- Blue blocks identify LLM-based components: Actor LLM, Teacher LLM, Ref LLM, Critic LLM, and Reward LLM. The reward-source switch changes the evaluator in RFT, PPO, GRPO, and CISPO between Reward LLM and a non-neural Verifier.
- Objectives are shown without line breaks inside expressions. Independent definitions appear on separate lines.

Cards are created once. Expanding, collapsing, and highlighting update existing DOM nodes without replacing, moving, hiding, or dimming other methods.

## Scope

The horizontal axis distinguishes external responses from current-policy rollouts. Offline means fixed data, not necessarily every off-policy setting.

The vertical axis distinguishes reward-based policy gradients from direct token, preference, or KL matching. On-policy hard distillation uses teacher token labels on student-generated prefixes; on-policy soft distillation uses direct KL matching (GKD). Both detach the sampling step. Policy-gradient distillation is a different variant.

PPO, GRPO, and CISPO use recent rollout snapshots and importance corrections. Their on-policy placement describes their common data-collection loop, not equality between the sampling and trainable policies at every update.

Hard distillation shares token cross-entropy with SFT and RFT; soft distillation matches teacher distributions with KL. Each has off-policy and on-policy cards, differing in prefix source. PPO and GRPO share the clipped surrogate, while GRPO and CISPO share group-relative advantages. CISPO clips detached importance weights instead of the surrogate objective and omits the reference KL penalty in the original recipe.

## Development

GitHub Pages publishes the repository root from the `main` branch. Push changes to `main` to update the live site.

Serve the source directory:

    python3 -m http.server 8054 --bind 127.0.0.1

Rebuild the portable HTML:

    python3 build.py

KaTeX 0.16.22 is vendored under its MIT license.
