# Cover letter — IEEE Transactions on Biometrics, Behavior, and Identity Science

[Date]

Dear Professor Nixon,

Please consider the enclosed manuscript, **"How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification"**, for publication in *IEEE Transactions on Biometrics, Behavior, and Identity Science* as a **Regular Paper**.

## 1. Fit to the journal's scope

The manuscript is an evaluation and protocol study, and we submit it as one. T-BIOM's broad topics include "standards for their deployment", "novel (challenging) biometric databases, protocol and benchmarking, large scale evaluation, confidence interval estimation, and performance modeling and prediction", together with "person re-identification" and "detection, tracking, identification, and de-identification in surveillance". The manuscript contributes to the second and third of these, on a surveillance identity problem. Its central artefact is a protocol, not a network: three label-free site probes that decide which identity functions a camera network can support before installation, and five delivery tiers that each carry an acceptance condition, a measured benefit and a stated failure mode.

## 2. The problem

Re-identification models report above 90% mAP on benchmarks. An integrator who has just installed six cameras in an office cannot act on that number, because the benchmark's conditions — tight upright crops, viewpoints shared with training, no seated subject at 50 px — are not the conditions of a building. The question that decides the project is which identity functions the network can support, and whether that can be established before the contract is signed.

## 3. What the manuscript reports

We freeze an off-the-shelf OSNet, never tune it on the site, and measure where a learned re-identification representation stops being informative.

Four results, each with a population attached:

1. **Non-composability (Section VII).** A local constraint applied to a global partition damages correct merges it never addressed. Three independent demonstrations: deleting adjudicated-wrong edges splits zero groups; constrained minimum colouring honours all 27 adjudicated labels yet moves the identity count from 443 to 450; a verifier with pairwise AUC 0.833 destroys three of five known-correct merges, monotonically in its threshold. This is why appearance signals belong in the ordering of human review, not in the decision.
2. **Three probes (Section V).** Face pixel budget, regional co-visibility, wardrobe diversity — each measurable in under a day, each gating one tier. We state explicitly which part is new: the 60 px boundary is inherited from GB/T 31488-2015 and the clothing statistic is a class-size count; what is new is P2's *unit of answer* — co-visibility is a property of a **region**, not of a camera pair — which reverses the table an integrator would naturally write (the pair declared the only overlap reaches AUC 0.904; pairs declared disjoint reach 0.231).
3. **A tiered delivery ladder with measured benefits (Section VI).** 42.3% same-camera recall across time; AUC 0.904 (frozen OSNet) and 0.994 (independent encoder) inside the one genuinely co-visible pair; an attribute-consistency gate that lifts merge precision from 0.730 to 0.930 at unchanged recall; pose-conditioned templates that lift selector recall from 56.4% to 68.2% with **no face at all**, and to 69.8% with a measured face channel — against an 83.2% identity-oracle ceiling, which we report as a ceiling and not as a result; a 27-item adjudication queue that resolves in 20 to 30 minutes.
4. **Acceptance numbers do not survive a capture change (Section VIII).** Across three further capture days of the same pipeline, coverage moves by 0.001 while merge error doubles, and the gallery that covers its own day best transfers worst. The statistics an integrator would naturally accept on do not see this, and one camera-day that scored perfectly on all three turned out to be the one whose tracking had failed.

Every headline number is reported with its population and interval. The gallery-membership truth layer was re-annotated blind by a second person (257 tracks, Cohen's κ = 0.925, 95% CI 0.891–0.958). The pair layer (153 pairs) was judged by one person, and we state this as the study's most serious limitation rather than leave it implicit.

## 4. Datasets and reproducibility

The main results rest on video that **cannot be released**: it records identifiable individuals on a client site under an authorisation that forbids redistribution. We state this in the manuscript's Data Availability section and justify it here, as the author instructions require.

What is released with the article, and provided as separate supplementary material, is everything needed to reproduce the analysis from track tables and embeddings: the protocol specification, the three probe thresholds and their calibration, the structure and sampling mechanism of both truth sets, the adjudication rules, every result file cited in the article, and the scripts that generate them (Supplementary Appendix S1). No training is involved, and every experiment is deterministic given the track tables and embeddings. Reviewers who need the underlying media for independent verification may obtain access under a data-use agreement; we will arrange this promptly if the editors or reviewers request it.

## 5. Declarations

The manuscript is original, has not been published, and is not under consideration elsewhere. It is not an extension of a conference paper. All authors have approved the submission and the author order, and declare no competing interests. [If an arXiv preprint exists, state it here with the identifier.]

Suggested reviewers can be supplied on request; we will name five, none from our institutions and none with a joint publication in the past three years.

Yours sincerely,

[Name], on behalf of all authors
[Department, Institution]
[E-mail] · ORCID: [0000-0000-0000-0000]
