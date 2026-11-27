# Research findings

Snapshot through 2026-11-27. Palimpsa compares SymA attention with a Llama-style Base at about 21.7M total parameters. The experiments below use one training seed per condition. Paired confidence intervals describe variation across the tested examples or layouts. Seed-to-seed variation remains unmeasured. The repository includes the [aggregate measurements](../results/research_2026-11-27.json) and the [training and dialogue summary](../results/2026-11-27_aggregate.json).

## Language modeling and downstream tasks

We ran a learning-rate sweep, attention ablations, and paired Base and SymA training at the same token budget. At 0.435B tokens, adding the exact-match-length bias improved validation NLL in two settings. A3 improved on Base with RoPE, and SymA improved on the forget-gate model A1. A2, which combines RoPE with both SymA biases, had the lowest validation macro NLL among the tested variants at that budget.

At 1.30B tokens, macro validation NLL was 2.6122 for Base and 2.5907 for SymA. SymA had lower NLL in web, code, and math. The comparison matches tokens. SymA processed fewer tokens per second in the measured training runs, so the ranking at equal compute remains open.

The matched 1.30B checkpoints had mixed zero-shot results. For example, raw SciQ accuracy was 70.20% for Base and 72.60% for SymA, while WinoGrande was 51.85% and 50.43%. On the first 512 LAMBADA examples, Base scored 17.38% and SymA scored 13.67%. The suite used candidate likelihood, and LAMBADA used exact target prediction. The measured NLL gain has yet to translate into a consistent downstream advantage.

## Binding and exact token matches

The binding test swaps two values in a table while keeping the query and the eight candidate values fixed. A pair counts as correct when the model selects the right value in both halves.

| Variant at 0.435B | Both halves correct |
|---|---:|
| Base | 10.9% |
| A1, forget gate | 18.7% |
| SymA, match bias and forget gate | 15.3% |
| A2, match bias, forget gate and RoPE | 24.8% |
| A3, match bias and RoPE | 35.1% |

Exact binding was sensitive to the architecture and training stage. SymA scored 29.5% at an intermediate 824M-token checkpoint and 9.4% at 1.30B. Base also fell from 10.9% at 0.435B to 5.8% at 1.30B. The short and long endpoints follow different LR decay schedules. In the remapping control, the late SymA checkpoint more often retained its preferred word when it moved to a different record.

We varied the strength of the match bias and swapped its learned tables between SymA checkpoints. The early tables recovered a small part of the late checkpoint's exact binding score. Swapping only layers 2 and 5 raised exact accuracy from 9.4% to 13.0% on new layouts. Both versions scored 0.7% on the tested paraphrase. This intervention establishes a contribution from the two tables to the measured readout. The remaining weights and their interaction with the tables also matter.

## Transfer beyond the exact query

- **Change the format.** A3's exact advantage survived new keys and a switch to colons when the record and the query used the same separator. It disappeared when the separators differed. In that condition the final match row was zero, and A3 trailed Base by 1.40 percentage points on the new vocabulary. On matching formats, disabling the match bias at the final query position removed most of A3's advantage.
- **Vary the training data.** In a balanced short pilot, independent separators improved A3's mean accuracy on two held-out query forms by 4.30 points and reduced Base's by 3.20 points. The best absolute score was 18.05% for Base trained with matching separators, driven mainly by the statement form. All four fine-tuned branches had worse language NLL than their parents. With a fixed table and a changed queried key, the strongest Base scored 21.1% on the near statement ending in `is` and 0.1% on the far statement ending in `is`. The test moves words as well as changing distance.
- **Mix binding examples into language modeling.** Two short continuations from late SymA used 5% contextual examples. The first used ordinary cross-entropy. The second increased the answer-token weight eightfold and retained the ordinary language objective elsewhere. Both missed their predefined binding and language NLL criteria. These results apply to the tested training budget and data mixture.
- **Probe the readout.** In late SymA, a weak correct-value logit preference appeared just after the key and faded through the two long queries. Post-block activation swaps at the key or across the whole query had little effect on the final preference on those forms. The exact `key =` control showed a much stronger response at the final query position. These interventions measure the effect of replacing selected activations. The location and form of any internal representation remain open.
- **Repeat a known key.** Appending the correct `key =` to the long query increased paired accuracy on fresh tables. Supplying a different key drove original-query pair accuracy to zero in all three tested models. The experimenter supplied the key, so the result measures the effect of a retrieval cue under known-key conditions.

## Dialogue fine-tuning

We gave the 1.30B Base and SymA checkpoints the same training examples, order, and assistant-token objective. After SFT, SymA answered 18 of 64 local English instruction items exactly and Base answered 2. Base answered 2 of 80 GSM8K items exactly, and SymA answered none. On the restricted Python tasks Base passed 48 of 64 and SymA passed 45. Both scored zero on the 16 tasks from unseen expression families. Both also had higher language-modeling NLL after SFT.

The dialogue pilot provides preliminary evidence for improved instruction following in this specific prompt format. The ordinary language checkpoints give the lower NLL. The results leave broader dialogue quality, mathematical reasoning, code generalization, and training-data overlap unresolved.
