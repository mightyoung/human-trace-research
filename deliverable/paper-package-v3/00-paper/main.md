# How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification

**Authors.** [Given name] [Family name] (a,\*), [Given name] [Family name] (a), [Given name] [Family name] (b)

(a) [Department], [Institution], [City], [Postcode], [Country]

(b) [Department], [Institution], [City], [Postcode], [Country]

\* Corresponding author. E-mail: [address]. ORCID: [0000-0000-0000-0000]

*Article type: Full Length Article. Submitted to Neurocomputing.*

---

## Abstract

Cross-camera re-identification is reported above 90% mAP on benchmarks, yet that number does not tell an integrator which identity functions a new camera network can support. We answer with a deployment-time measurement procedure. From a six-camera, 35-minute capture of an operating office floor (1,574 tracks, 443 hypothesised identities) and an eight-segment corpus with 26 named identities, we reduce availability to three probes measurable in one working day: the face pixel budget, regional co-visibility between camera pairs, and wardrobe diversity. Each probe gates one delivery tier. Its five tiers each carry an acceptance condition and a measured benefit: same-camera identity across time (42.3% recall); cross-camera identity inside a co-visible region (AUC 0.904 for a frozen OSNet, 0.994 for an independent encoder); an attribute-consistency gate that raises merge precision from 0.730 to 0.930 at unchanged recall, stable over 63 leave-one-out folds; pose-conditioned templates that raise selector recall from 56.4% to 79.3%, 11.8 points without a face; and a 27-item adjudication queue resolvable in 20 to 30 minutes, satisfying every adjudicated constraint. Outside that one pair, cross-camera appearance sits at chance across six encoder and alignment configurations (AUC 0.450 to 0.622, every interval covering 0.5), and the face channel is unusable below 60 px (AUC 0.352 against 0.847). The tiers cannot be collapsed: three experiments show that a local constraint on a global partition damages correct merges it never addressed, and a verifier with pairwise AUC 0.833 destroys three of five known-correct merges. The binding constraint is capture geometry, not the encoder.

**Keywords:** Cross-camera re-identification; Deployment protocol; Selective prediction; Human-in-the-loop; Co-visibility; Field study

---

## 1 Introduction

A vendor demonstrating cross-camera ReID shows two crops of the same person and a similarity score. A facilities manager signing the contract needs a different object: a statement of the form "with this network, you will be able to answer *this* class of question, at *this* accuracy, with *this* much operator time". Between those two objects lies the entire deployment problem, and benchmark mAP does not address it, because the benchmark's operating conditions (tight upright walking crops, query/gallery splits that share viewpoints with training, no seated subjects at 50 px) are not the conditions of an office.

This paper constructs the second object. We propose no new representation: the encoder is off-the-shelf, frozen, and never tuned on the site. The question we answer is:

> **Given a fixed camera network and a frozen encoder, what identity functions can an integrator legitimately promise, and how can the integrator know this before deployment?**

Our answer has three parts, each of which can be checked against measurements.

**First, the question decomposes into three site measurements.** Whether cross-camera identity is available depends on quantities that have nothing to do with model choice: how many pixels a face occupies (P1), whether two cameras can see the same region at all (P2), and how repetitive the population's clothing is (P3). Each probe costs a day of measurement and needs no labels, and each maps to a specific delivery tier. Section 5 defines them, calibrates their decision boundaries on our data, and shows that together they predict, per camera pair, whether appearance association will work: the pair our probes certify as co-visible reaches AUC 0.904, and a pair they certify as disjoint sits at 0.231, an inversion strong enough to serve as a warning.

**Second, the deliverable is a ladder.** We define four tiers, L0 same-camera across time, L1 co-visible region, L2 attribute-consistent merging and L3 human adjudication, where each tier states an acceptance condition and carries a measured benefit: 42.3% recall at L0, AUC 0.904 to 0.994 at L1, merge precision 0.730 → 0.930 at L2, and a 27-item queue adjudicable in 20 to 30 minutes at L3 that satisfies all 27 adjudicated constraints at once. A fifth, L2+, uses pose-conditioned templates and raises selector recall from 56.4% to 79.3%, more than half of which needs no face. The tiers are ordered by evidence strength, and an integrator buys the highest tier its probes and its operator budget can support. Section 6 specifies them.

**Third, the ladder cannot be collapsed.** The obvious economy is to replace L3 with an automatic verifier: score every merged pair, cut below a threshold. Three experiments show that any local constraint on a global partition damages correct merges it never addressed. Deleting the fifteen adjudicated-wrong edges splits zero groups. Constrained minimum colouring honours all 27 labels and moves the identity count from 443 to 450, because a 32-track group constrained by a single pair can shed exactly one track. A verifier whose *pairwise* AUC is 0.833 removes 3 of 5 known-correct merges, monotonically in its threshold. Those three failures are why the protocol is tiered, and why appearance signals belong in the ordering of human review rather than in the decision. Section 7 gives the three instantiations.

Work on cross-camera identity is largely diagnostic: appearance association degrades in unconstrained capture, temporal priors are unreliable when clocks or topologies are uncertain, and post-hoc verification is fragile. All three hold at this site, and Sections 5 to 7 measure them. A diagnosis on its own still leaves an integrator with nothing to buy, so the probes and tiers below are built as the object that can be bought, with each failure mode appearing at the point where it forces a design decision.

### 1.1 Contributions

1. **A three-probe deployment diagnostic** (Section 5). Three label-free site measurements (face pixel budget, regional co-visibility, wardrobe diversity), each with a calibrated decision boundary, jointly predicting which appearance-dependent tiers a site can support. Boundary anchors: 60 px (AUC 0.352 → 0.847), one co-visible pair out of fifteen (AUC 0.904 vs 0.231), and 35 clothing combinations over 673 people, with the modal combination worn by 118.
2. **A tiered delivery protocol with measured rungs** (Section 6). L0–L3 plus L2+, each with an acceptance condition, a measured benefit, and a stated failure mode. The protocol converts a binary "does ReID work" into a menu an integrator can price.
3. **A non-composability result that motivates the tiering** (Section 7). Three independent demonstrations that local constraints do not compose over a global partition, including the corollary that pairwise AUC does not predict partition-level utility. This is the formal reason the ladder cannot be replaced by an automatic stage.
4. **A pair-level truth set and reporting discipline** (Section 4). 153 adjudicated cross-camera pairs over 12 camera pairs, in two deliberately separate populations, with an explicit statement of which numbers may not be compared with which.
5. **An operational artefact and its measured cost** (Section 8). An offline, single-file adjudication surface, JSON write-back as must-link/must-differ constraints, and a measured 20–30 minutes for 27 items.

The scope is one site, one day and one adjudicator (Section 10). We publish the protocol and the parameters, not the video.

---

## 2 Related Work

### 2.1 Representations and camera bias

Supervised ReID matured on Market-1501 [1] and MSMT17 [2] with strong backbones and part-based or transformer aggregation [3, 4, 5]. OSNet's omni-scale features remain the standard frozen extractor in deployment settings. Two properties of that literature matter here. First, reported gains are measured under the same capture conditions as training: tight boxes, walking subjects, and query/gallery splits that share camera viewpoints with training. Second, camera identity is a known confound: models latch onto per-camera colour and background statistics, which motivated camera-aware and camera-adversarial training [6, 7, 8]. Our study inherits the diagnosis and changes the response. We measure the operating envelope of a frozen encoder and turn it into a routing rule: appearance is used where the probe says it is informative, and elsewhere the protocol falls back to geometry, semantics, or a human.

### 2.2 Clustering-based identity graphs

Unsupervised and pseudo-label pipelines build identities by clustering tracklet features, typically with DBSCAN or agglomerative linkage [6, 9], and camera-aware variants forbid merges within a camera. Our deployed graph follows this pattern: within-camera average linkage, then cross-camera greedy merging under a similarity threshold, a same-frame co-occurrence block, and an attribute gate. The relevant prior for Section 7 is that these pipelines are all *partition* algorithms over a transitive closure, so a single edge decision propagates. Existing work treats the resulting sensitivity as a threshold-tuning problem. We show it is structural and that it governs post-hoc repair as well, which is what makes a tiered protocol necessary.

### 2.3 Spatio-temporal constraints and topology

Spatio-temporal priors are usually introduced as transition-time models: a person cannot be in two places at once, and transit between cameras takes at least τ seconds [10, 11]. These priors assume a synchronised clock and a known notion of which cameras can co-observe a region. Multi-camera tracking benchmarks [12] make exactly these two assumptions, and we show how the second one fails in a plain office. With an OSD-based absolute clock the constraint does surface 27 contradictions that similarity never ranks, but the co-visibility table those contradictions are judged against is itself wrong at pair granularity, because co-visibility is a property of regions. We therefore use the audit as a trigger that selects review items, and report that a hard temporal gate blocks one merge, which is why it appears in the protocol as an item selector and never as a criterion.

### 2.4 Selective prediction and risk control

Rejecting rather than answering is well studied as selective classification [13], and risk-controlling calibration (Learn-Then-Test) converts a heuristic operating point into a finite-sample guarantee [14]. In an auto-built gallery the object to reject is the *cluster*, not the query [15]. Our protocol is a deployment-level instance of the same idea: L3 is an explicit abstention tier, and the two-probe report of Section 6.6 (precision on covered pairs *and* the fraction of audited groups that turn out mixed) is the selective-risk statement an operator needs. The difference from prior work is that we also quantify what abstention *costs* (20–30 minutes per 27 items) and show why the abstention cannot be automated away (Section 7).

### 2.5 Field studies and human-in-the-loop identity

Deployment studies in surveillance consistently report that fully automatic cross-camera identity is not achieved, and that operator review is where the system earns its value [16, 17, 18, 19]. We are not aware of a study that (i) reduces the availability question to a small set of pre-deployment probes, (ii) attributes the appearance ceiling to a measurable pixel budget rather than to representation quality, and (iii) turns the result into a tiered protocol with per-tier acceptance conditions. That combination is this paper.

---

## 3 Two Evidence Domains

The protocol is developed on one domain and validated on a second, because the tiers need different kinds of evidence: deployment-scale structure from D1, and named-identity ground truth from D2. Section 4 explains why the two are never pooled.

### 3.1 D1: six-camera, 35-minute deployment

One floor of an office building: an open work area, a tea room / small meeting room, a glass-walled corridor, and a large multi-purpose space. Six cameras record continuously for 35 minutes (2,099 s) at 25 fps. Four are 2560×1440 and two (318, 323) are 1920×1080. No calibration, no depth sensor, no face turnstile. Fields of view do not overlap except for one pair.

**Table 1. D1 capture and graph statistics.**

| Item | Value |
|---|---|
| Cameras | 6 (305, 316, 318, 320, 323, 343) |
| Resolution | 4 × 2560×1440, 2 × 1920×1080 |
| Duration / frame rate | 2,099 s / 25 fps |
| Tracks produced | 1,894 |
| Tracks entering the identity graph | 1,574 |
| Hypothesised identities (graph v0) | 443 |
| Cross-camera groups | 56 |
| Largest group | 67 tracks |
| Usable face embeddings | 150 / 1,894 tracks (7.9%) |
| Median face width | 52 px |

### 3.2 D2: eight-segment named-identity corpus

A second capture of the same floor, split into eight segments, with 445 named tracks over 26 identities (14 tracks per identity at the median, 62 at the maximum). D2 is the only domain where the identity of every track is known, which is what the selector and template-pool experiments in Section 6.4 require. D2 has a different detector/tracker configuration from D1 and is therefore never used for merge-precision claims.

### 3.3 Clock discipline

Every temporal statement in this paper uses the on-screen display (OSD) absolute clock, never `frame / fps`. The six D1 streams start at different wall-clock moments; their frame-0 offsets relative to camera 318 are 0, +1, +2, +16, +25 and +121 s. Using per-stream frame indices silently converts a genuine co-occurrence into a handover, a mistake we made once. Correcting it raised measured true-pair recall in the merge comparison from 5/34 to 7/34 (Table 10) while leaving the identity count unchanged, which is the signature of a measurement fix and not of a modelling change. We recommend stating the clock convention in any multi-camera paper that reports temporal constraints; it costs one sentence and prevents a systematic error.

### 3.4 Pipeline

Detection and tracking run per camera (YOLO-family detector, ByteTrack [20] with a lowered activation threshold). We deliberately keep the tracker plain and did not add the association refinements of the StrongSORT family [21], so that each tier's behaviour is attributable to the site's geometry and to the frozen encoder rather than to tracking heuristics. Each track is encoded by OSNet-x1.0 [3] (MSMT17 [2] weights, frozen, 512-d mean over 8 temporally uniform frames). The identity graph is built in two stages: within-camera agglomerative clustering at cosine distance 0.15, then cross-camera greedy merging of cluster representatives at cosine ≥ 0.75, subject to three gates: a same-frame co-occurrence block (a merged entity may not be simultaneously visible in more than two cameras), a camera-overlap block, and, in the shipped version, an attribute-agreement gate requiring ≥ 8 of 9 clothing attributes to agree. The result is 443 identities covering 1,574 tracks (Figure 1).

![Figure 1](figs/fig1_pipeline.svg)

**Figure 1. The deployed pipeline and the three probe points.** Stage 1 is off-the-shelf. Stage 2 produces a transitive closure, which is what makes the non-composability result of Section 7 possible and what forces the protocol to be tiered rather than automatic. The probe icons mark where the diagnostic of Section 5 reads the site: pixel budget on the encoder input, co-visibility on the camera graph, wardrobe diversity on the attribute head.

### 3.5 What we deliberately did not train

We did not train or fine-tune the encoder, by design. A domain-adaptive fine-tuning attempt with leave-identity-out evaluation reduced operating-point recall from 24.2% to 11.0%, and a closed-set linear discriminant reached 0.986 on seen identities but 0.595/0.554 on held-out identities, which is the usual collapse of in-domain metrics and which we reproduced twice. A frozen encoder makes every number in this paper attributable to the site rather than to a fitting procedure, which is precisely what a diagnostic requires: the probes must predict the behaviour of a *given* encoder, not of a model that was tuned on the very site being measured.

---

## 4 Evaluation Protocol

### 4.1 Two truth sets, two sampling mechanisms

**T1: prior-sampled pairs (126).** Pairs drawn to test the merge rule as previously configured: 34 same-person and 92 different-person pairs over 7 camera pairs. T1 is biased toward pairs the existing system found plausible; it is a test set for the *rule*, not an unbiased sample of the population. It is the right set for measuring whether a gate preserves recall.

**T2: contradiction-triggered pairs (27).** Pairs surfaced by the temporal audit of Section 6.5, i.e. pairs the system had already merged and that a physical constraint says cannot be the same person. Twelve are same-person and fifteen different-person. T2 is biased toward *errors*, which is what makes it the right set for measuring whether a repair respects constraints.

Together, 153 pairs over 12 camera pairs. T1 and T2 are never pooled into a single precision figure. Doing so is a category error we made once and now forbid in our own reporting.

### 4.2 Three levels of granularity

A graph can be scored at three levels, and they disagree. Naming them is what makes the tiered protocol reportable. We note that multi-camera tracking metrics such as HOTA [22] and ID-based measures [10] score trajectories over time; our unit of analysis is the merge decision, which is why we score pairs and groups rather than trajectories.

- **Pair level.** A labelled pair is correct if its two tracks land in the same identity group (same-person label) or different groups (different-person label). This is what a merge rule consumes.
- **Group level.** A group is *mixed* if it contains at least one adjudicated different-person pair that was merged into it, and *pure* otherwise. This is what an operator consumes.
- **Cluster purity by inspection.** A group is inspected visually; this is what a reviewer consumes, and it detects failure modes (a group of shadows, a hat mistaken for a person) that pair labels never touch.

T2 yields 55.6% pair-level agreement (15 of 27 contradictions confirmed), but the operationally meaningful number is that **7 of the 15 touched groups are mixed and 8 are pure**. The same data supports two denominators and two different decisions, so the protocol reports both everywhere.

### 4.3 Reporting rules

Four rules, each adopted after getting it wrong.

1. **Never compare across detection versions.** D1 uses one detector/tracker configuration throughout. An earlier eight-segment study used a different one, and the cluster-mixing rate differed by a factor of six; cross-version comparisons were withdrawn. This is why D1 and D2 carry separate claims.
2. **Always state the denominator.** Pair-level precision over ground-truth-covered pairs and over suspicious pairs are different populations.
3. **Never let a similarity score stand as evidence of purity.** Cosine similarity cannot certify that a cluster is one person, and cannot refute a human adjudication either.
4. **State the truth set's sampling mechanism**, because every aggregate inherits it.

---

## 5 The Three-Probe Diagnostic

Availability of appearance-based identity at a site is predictable from three measurements that require no identity labels and no model training, and each measurement gates a specific tier of Section 6.

### 5.1 Probe P1: the face pixel budget

**Measurement.** Draw a random sample of tracks (we use 200), run a face detector inside each person box, and record the distribution of face-box widths together with the fraction of tracks that yield any face at all. Cost: under an hour with existing tooling.

**Calibration.** On D1, 150 of 1,894 tracks (7.9%) yield a face embedding, and median face width is 52 px. Splitting at 60 px gives a sharp transition. The cut is not ours: GB/T 31488-2015 [23] requires an inter-pupil distance of at least 60 px for recognition-grade capture.

- faces below 60 px: **AUC 0.352** (worse than chance; the embedding is measuring compression artefacts)
- faces at or above 60 px: **AUC 0.847**

**Reading.** P1 answers a single question: *is the face channel available at all?* The boundary is a physical resolution threshold that the standard already requires, and we did not fit it to our data. A site whose median is below 60 px should not buy a face-dependent tier. On D1 the median is 52 px, i.e. below the threshold, and a fixed 60 px on a 1440p frame corresponds to a person roughly 5 m from the lens while most of this floor's seated occupants are further than that from every camera. The remedy is physical: reposition or add cameras to reach a median above 80 px.

![Figure 2](figs/fig2_pixel_budget.svg)

**Figure 2. The pixel budget.** Left: face-based verification on this site splits at 60 px into chance-level (0.352) and usable (0.847). Right: the floor's median face width (52 px) sits *below* the recognition threshold, which is why the embedding measures compression artefacts rather than identity.

P1 predicts failure *before* deployment and names the fix. It does not depend on which face model is used, because the transition is a property of the pixel budget and not of the encoder.

### 5.2 Probe P2: regional co-visibility

**Measurement.** For each camera pair, decide whether the two fields of view share a *region*, and if so which. The cheap version is a site walk-through with a shared marker; the reproducible version is a homography test on a common planar patch, or a statistic over simultaneous detections. Cost: half a day for six cameras.

What matters is the *unit of the answer*. A camera pair is not a binary predicate. On D1 the site survey declared fifteen camera pairs disjoint and one pair (318–320) overlapping. Both parts of that table are wrong in a specific way: 318–320 is genuinely co-visible (tea room seen from two sides, homography AUC 1.000), but so are regions of pairs the survey called disjoint, and the consequence shows up directly in the identity graph.

**Aggregate calibration.** We first evaluate six feature configurations on T1's cross-camera pairs: two representations (OSNet-x1.0, and an image-embedding model we denote E2, covering 1,322/1,574 tracks) crossed with three alignment schemes: raw cosine, per-camera mean subtraction ("cc"), and per-camera PCA whitening to 33 dimensions ("cw", one third of the per-camera track count, following our earlier dimensionality finding).

**Table 2. Cross-camera discrimination on T1 (34 same / 80 different pairs; E2 covers 28/61).**

| Representation | Alignment | Same (median) | Diff (median) | AUC | 95% CI |
|---|---|---|---|---|---|
| OSNet | raw (deployed) | 0.499 | 0.532 | **0.490** | [0.350, 0.625] |
| OSNet | per-camera mean-sub | 0.022 | 0.015 | 0.531 | [0.403, 0.666] |
| OSNet | per-camera whitening | 0.016 | −0.011 | 0.511 | [0.397, 0.626] |
| E2 | raw | 0.759 | 0.777 | **0.499** | [0.330, 0.658] |
| E2 | per-camera mean-sub | 0.160 | 0.029 | **0.622** | [0.470, 0.765] |
| E2 | per-camera whitening | −0.052 | 0.012 | 0.450 | [0.321, 0.585] |

Two observations shape the probe. First, **no configuration separates same-person from different-person cross-camera pairs**: all six confidence intervals contain 0.5. For the deployed encoder the same-person median (0.499) is *below* the different-person median (0.532), so the ordering is inverted rather than merely weak. Second, mean subtraction is the only scheme that moves the point estimate (E2: 0.499 → 0.622), and it does so by collapsing both distributions onto a near-zero band whose ordering is fragile: at a 2% false-accept rate it recovers 7 of 28 true pairs, at 10% it recovers 13 of 28. That is a better *ranking*, not a usable *threshold*.

An aggregate at chance would normally end the diagnostic, with appearance declared unusable. P2 shows that the aggregate is the wrong statistic.

**Pair-level calibration.** Stratifying the same data by camera pair dissolves the aggregate:

**Table 3. Per-pair AUC on T1 (OSNet raw; pairs with both classes).**

| Camera pair | same / diff | AUC | Probe P2 reading |
|---|---|---|---|
| **318–320** | 11 / 52 | **0.904** | co-visible region confirmed (homography AUC 1.000); E2 reaches **0.994** |
| 316–323 | 2 / 5 | 0.600 | n too small to interpret |
| 318–323 | 3 / 4 | 0.250 | no shared region; inverted |
| **316–343** | 13 / 18 | **0.231** | no shared region; strongly inverted |
| 323–343 | 1 / 1 | 0.000 | n = 2, not interpretable |
| 305–316, 316–318 | 2 / 0, 2 / 0 | n/a | no negatives; AUC undefined |

The pattern is not "some pairs are harder". It is that **one pair carries essentially all of the signal, and it is exactly the pair the probe certifies as co-visible**. Conversely 316–343, with the largest sample (31 pairs), is anti-correlated: the model ranks strangers above the same person. This is why P2 must be measured at region granularity before any threshold is chosen: a threshold tuned on 318–320 is actively harmful on 316–343, and that is the mechanism behind our earlier observation that lowering the global threshold triples the merge count while true-pair recall stays at 7/34.

**What P2 buys.** It converts an unanswerable question ("does cross-camera ReID work?") into an answerable one per pair ("does this pair share pixels' worth of person?"), and it tells the integrator where to put the overlap when the site is re-cabled. It also explains the 815 s finding of Section 6.5 without appeal to model behaviour: two cameras on opposite sides of an open floor can both see one workstation, so a person seated there is genuinely co-visible for minutes.

### 5.3 Probe P3: wardrobe diversity

**Measurement.** Over a day of detections, count distinct clothing combinations and the size of the largest equivalence class. Cost: hours, using the attribute head that is already part of the pipeline.

**Calibration.** On D1, 673 people occupy only **35 distinct clothing combinations**, averaging 19.2 people per combination; the modal one, a blue upper with long sleeves and a black lower, is worn by **118 people**.

**Reading.** P3 decides the *direction* in which clothing evidence may be used, and the direction is asymmetric. "Different clothes" is a valid veto: two entities whose predicted attributes disagree on two or more fields are very unlikely to be the same person in a population where the modal outfit has 118 wearers. "Same clothes" is not a vote: with 118 people in one outfit, agreement carries almost no identity information. A site with uniforms has the opposite failure: everyone agrees, so the attribute channel becomes uninformative instead of misleading, which is why P3 reports the size of the modal class and not a single diversity index.

This asymmetry is what makes L2 (Section 6.3) a *selective* repair: it removes merges without touching recall, because it only ever acts as a veto.

### 5.4 The routing rule

The three probes compose into a routing rule that an integrator can execute before signing a contract:

```
P1  median face width
    80 px or more  -> face tiers available (L2+)
    60-80 px       -> face on a minority; bonus only
    below 60 px    -> no face tier; spend on operator time

P2  shared region per camera pair
    yes            -> L1 available (expect AUC 0.90+)
    no             -> L1 unavailable; use L0 + L3

P3  modal clothing class
    below ~5%      -> veto informative; use L2
    above ~20%     -> veto weak; L2 optional
```

Three properties make it a diagnostic. Each decision boundary is physical (P1: resolution), geometric (P2: shared pixels) or statistical (P3: class size), so it does not drift with model choice. Each probe is measured on the site itself, so it cannot be inherited from a benchmark. Each probe maps to a *specific* tier, so the output is a menu and not a score.

![Figure 3](figs/fig3_protocol.svg)

**Figure 3. The protocol at a glance.** Three probes on the left, five tiers on the right. The staircase encodes the trade-off: higher tiers deliver more identity but are available at fewer sites and depend on stronger preconditions. L0 needs no probe; L1 is gated by regional co-visibility; L2 by wardrobe diversity; L2+ benefits from the face budget without requiring it; L3 needs only an operator budget. Because the ordering follows evidence strength, the tiers cannot be reordered or merged.

### 5.5 Validation of the rule on our two domains

The rule makes falsifiable predictions, and both domains confirm them. On D1, P2 predicts that only 318–320 should support L1: measured AUC 0.904 (OSNet) and 0.994 (E2), against 0.231 for the pair predicted disjoint. P1 predicts that the face channel should be unavailable: measured 7.9% usable and median 52 px, with the sub-60 px stratum at chance. P3 predicts that clothing should work as a veto but not as a vote: measured merge precision 0.730 → 0.930 with recall unchanged (Table 5), i.e. a pure precision gain, which is the signature of a veto.

On D2, where identities are known, P1 again predicts the sign of the effect and its size: usable faces cover 6.5% of the full 1,178 tracks (12.1% of the longer named tracks), and the face-anchor channel's gain lands almost entirely on the minority of identities with cross-pose face coverage (Section 6.4).

Two limitations of the rule are stated in advance. The numerical boundaries (80 px, 5%, 20%) are calibrated on one site and one population; the *ordering* of the probes is what we expect to replicate, not the exact cut points. And P2 requires a judgement about regions, which is a survey task with its own error rate, the one measured in Section 6.5.

---

## 6 A Tiered Delivery Protocol

We now turn the probes into a contract (Figure 3). The tiers are ordered by evidence strength: lower tiers rest on geometry and therefore nearly always hold; higher tiers rest on appearance and therefore require probe support. An integrator buys the highest tier the probes and the operator budget allow.

**Table 4. The protocol.**

| Tier | Function delivered | Acceptance condition | Measured benefit (domain) | Failure mode if pushed beyond condition |
|---|---|---|---|---|
| **L0** | Same-camera identity across time | One camera; any resolution | 42.3% recall on same-camera cross-period pairs (D2) | Fragmentation, not confusion: 5.5 tracks per person in a segment (D2) |
| **L1** | Cross-camera identity inside a co-visible region | P2 certifies a shared region | AUC 0.904 (OSNet) / 0.994 (E2) (D1, pair 318–320) | Threshold tuned here is harmful elsewhere (0.231 on 316–343) |
| **L2** | Attribute-consistent merging | P3 certifies a small modal class | Merge precision 0.730 → 0.930 at unchanged recall (D1) | Uniforms make the gate uninformative |
| **L2+** | Pose-conditioned templates | ≥ 3 observable poses per identity | Selector recall 56.4% → 68.2% (no face) → 79.3% (D2) | Identity count too small to train a projection |
| **L3** | Human adjudication queue | Operator budget | 27 items / 20–30 min; 15/15 negative and 12/12 positive constraints honoured (D1) | Queue grows linearly with capture length |

### 6.1 L0: same-camera identity across time

The cheapest tier and the one an integrator can almost always buy. Within one camera, appearance is comparing like with like: same viewpoint, same lens, similar scale. On D2, the only domain with per-track names, 42.3% of same-camera cross-period pairs are resolved correctly at the deployed operating point. D1 supplies the deployment-scale structure (1,574 tracks, 443 identities) while D2 supplies identity-level recall, and this division is why the two domains are never pooled. The dominant error is fragmentation and not confusion (5.5 tracks per person within a segment, 63% of breaks longer than 10 s), which matters for the protocol because fragmentation is *recoverable by adding evidence over time* and confusion is not.

A quantitative caution that belongs to this tier: within a camera, appearance fails specifically when the same person changes *pose class*, not when they change location. On D2 the same person's box height spans a median of **3.76×** (maximum 11.6×) over a day, because a standing full-body view and a seated head-and-shoulders view share little body-part composition. At pair level, same-person similarity falls from a median 0.887 (comparable box height) to 0.557 (very different box height), and this holds in 14 of 14 identities with enough samples (median correlation −0.48). Different-person similarity is flat across the same bins (0.519–0.553), which is why raising the threshold does not help and why L2+ exists.

### 6.2 L1: cross-camera identity inside a co-visible region

Where P2 certifies a shared region, cross-camera appearance becomes ordinary same-viewpoint matching, and the tier delivers near-benchmark behaviour: AUC 0.904 for the frozen OSNet and 0.994 for a second, independent encoder on the same pair. The machinery works when the geometry permits it. The pair in question is a tea room seen from two sides, and the two views differ mainly in azimuth, which is the condition ReID benchmarks assume and which most office installations do not provide.

The tier's acceptance condition is therefore geometric, not statistical: buy L1 where you can buy overlap. Its design implication is stated in Section 9: 5–10% overlap between adjacent views is worth more than any encoder upgrade, because it moves a pair from the 0.231 regime to the 0.904 regime.

### 6.3 L2: the attribute-consistency gate

The only *automatic* component that earns its place is a *consistency* check, and P3's asymmetry is why. Nine clothing attributes (upper colour, lower colour, sleeve length, backpack, hand-carried object, hat, glasses, and two more) are predicted per track from the same frozen encoder. A candidate merge is accepted only if the two entities' predictions agree on at least 8 of 9 fields.

**Table 5. Effect of the attribute gate (D1, threshold 0.75, T1).**

| Configuration | Identities | Merges | T1 same | T1 diff | Merge precision (judged) |
|---|---|---|---|---|---|
| no gate | 434 | 76 | 5/34 | 0/80 | 0.730 (46/63) |
| attribute gate (≥8/9) | **443** | 67 | 5/34 | **0/80** | **0.930 (40/43)** |

The gate removes 9 merges, of which 6 were wrong, and adds 9 identities, a *selective* repair that leaves true-pair recall and false-merge count untouched. Three properties make it the right kind of signal for a protocol tier:

- It asks whether the same clothes are worn, not whether the vectors are close, so it does not inherit the camera bias that makes cosine unreliable across views.
- It is evaluated only on attributes with measured reliability. Five fields reach 0.88–0.99 agreement; carrying and backpack are capped at 0.656 and 0.458 and are excluded from the gate's decisive fields.
- Its operating point is stable. Leave-one-out over 63 folds selected ≥8 in every fold, so the gate is not a single-fold artefact.

For scale: raising the similarity threshold from 0.75 to 0.82 achieves roughly one fifth of this precision gain and loses recall. The gate is a different kind of evidence from a threshold. Its ceiling is stated as part of the tier: attribute *identity* is not evidence, so the gate is a veto, never a vote, and its benefit vanishes on a uniformed population.

### 6.4 L2+: pose-conditioned templates

The L0 caution becomes actionable here. Because same-person similarity collapses when pose class changes, the fix is a *template matched to the query's form*, since no threshold repairs a mismatch in body composition. Two mechanisms, measured on D2 (179 queries whose identity has a morning gallery entry in the same camera):

**Table 6. Selector recall at τ = 0.80 (D2, 179 queries).**

| Configuration | Recall | Δ | Face required |
|---|---|---|---|
| Gallery mean only (S0) | 56.4% | n/a | no |
| + box-height kernel weighting, σ = 0.6 (S1) | **68.2%** | +11.8 | **no** |
| + face-triggered real-time pool (S5) | **79.3%** | +11.1 over S1 | yes, on 31.7% of tracks |
| Upper bound: full-day material + kernel weighting | **92.7%** | +13.4 over S5 | n/a |

Three findings follow, and the first matters most in practice.

More than half of the gain needs no face at all. Weighting gallery templates by `exp(−|log h_query − log h_template| / 0.6)`, where `h` is box height, raises recall by 11.8 points with zero training and zero face detection. The mechanism is comparing against a template of the same body composition, not recognising the person better. The direct evidence: among pool hits where the gallery had failed, the used pool sample differs from the query in `|δh|` by a median of 0.06 against 1.05 for the gallery mean, and is closer to the query's form in 94% of cases. The pool is retrieving a sample at the *same scale*, not matching appearance better.

The face contributes coverage. The additional 11.1 points come from samples the gallery does not contain in any form, not from a stronger identity cue. The gain is concentrated where the gallery is structurally blind: at extreme pose difference (`|δh| ≥ 0.8`), recall goes from 7.0% to 88.4% (+35 of 41 total gains), while the 53 near-form queries move by +2 points. A site whose residents keep one posture will see almost none of this benefit, which is why the tier's acceptance condition is "≥ 3 observable poses per identity" and not "faces available".

The remaining 13.4 points come from material rather than method. The 92.7% upper bound is reached when the gallery contains samples of every form the query takes, and the gap from 79.3% is entirely the absence of such samples in an eight-segment corpus. That points to where to invest: more capture of the same person in more poses.

The tier's failure mode is worth stating because it recurs: training a pose-conditioned projection requires cross-pose pairs, and of 1,028 available pairs in D2, 799 (78%) belong to one identity, leaving only 13 of 26 identities with any cross-pose pair. Leave-identity-out evaluation therefore has insufficient data, which is the same limitation that ruled out encoder fine-tuning (Section 3.5). The inference-time kernel weighting avoids this entirely, which is why the tier ships in inference form.

### 6.5 L3: the human adjudication queue

L3 is a product of the protocol and not a concession. Its item selection follows from Section 5.2: a contradiction between the graph and the clock is a region-level co-visibility question that a person can answer from two crops, so it is exactly the right thing to put in front of a human.

- **Item selection.** Contradictions from the temporal audit (27 items on D1). Each item is a pair of tracks from one group that cannot physically be the same person *if* the co-visibility table were right, which is precisely the question the human resolves.
- **Presentation.** Three frames per side at *identical OSD instants*, cropped from native-resolution video with the target box drawn. Uniform temporal sampling was abandoned: in a seated-office scene it lands on monitors, and readable yield drops sharply. Instants are instead chosen to maximise the minimum box height across the two sides.
- **Delivery.** A single offline HTML file (2.05 MB, images embedded) that works on a phone with no network, because the adjudicator is not necessarily at a workstation.
- **Write-back.** Verdicts export as JSON and are ingested as must-link / must-differ constraints, never as a similarity threshold.
- **Explicit non-labels.** "Cannot see clearly", "walked past the boundary", "mixed identities on one side" are recorded but excluded from the truth set, because they do not constrain identity. Removing an item from the denominator is not the same as labelling it.

**Measured yield.** All 27 items were adjudicated by one person in 20–30 minutes: 12 same-person, 15 different-person, no items unreadable, 27/27 usable. Group level: 7 of the 15 touched groups are mixed and 8 are pure. Applied as constraints (Section 7.2), they take the identity count from 443 to 450 while satisfying 15/15 negative and 12/12 positive relations simultaneously. That is a strict improvement in constraint satisfaction for under 2% structural change, which is what 27 labels can buy in a 1,574-track graph.

### 6.6 The composite deliverable, and the two numbers that must travel together

Assembled, the protocol delivers the following statement for D1. Table 7 is the form in which an integrator can accept or reject a system.

**Table 7. Deliverable statement for D1.**

| Claim | Tier | Status | Evidence |
|---|---|---|---|
| Same-camera identity across time | L0 | Supported for 42.3% of same-camera cross-period pairs (D2) | §6.1 |
| Cross-camera identity, 318↔320 | L1 | Supported (AUC 0.904 / 0.994) | Table 3 |
| Cross-camera identity, other pairs | L1 | Not available; candidates with evidence only | Table 3, 316–343 at 0.231 |
| Merge precision (T1-covered pairs) | L2 | 0.930 with attribute gate | Table 5 |
| Merge precision (contradiction-triggered pairs) | n/a | 0.444 pair-level; 7/15 groups mixed | §6.5 |
| Selector recall, named identities | L2+ | 56.4% → 79.3% (ceiling 92.7%) | Table 6 |
| Face-based verification | P1 | Not available at this capture geometry | 7.9% usable, <60 px AUC 0.352 |
| Automatic repair of a merged group | n/a | Not available; L3 required | §7 |

The two precision rows are the ones most often conflated in practice, and the protocol requires that they travel together. A system that reports "0.930 precision" and omits "7 of 15 audited groups are mixed" has told the truth about one population and withheld it about the other. We therefore make the pair of numbers a reporting requirement of L2 and L3, not an optional disclosure.

### 6.7 Why L3 cannot be replaced by an automatic stage

The natural objection to a tiered protocol is that L3 is expensive and could be automated: train a verifier on the adjudicated pairs, apply it to every merge, and keep the high-confidence ones. Section 7 answers this objection with three experiments, but the short form belongs here because it determines the protocol's shape. Any operation that acts on one relation inside a partition changes a transitive closure, and a transitive closure has no obligation to preserve the merges that were correct. Consequently the choice is not "automatic or manual at the same accuracy"; it is "manual review of selected items, or automatic damage to unselected ones". That is why L3 is a tier and not a step.

---

## 7 Why the Tiers Cannot Be Collapsed

This section supplies the protocol's necessity argument. The situation it analyses is generic: a pipeline has produced a partition; someone has evidence that one relation inside it is wrong; the natural repair is to act on that relation. We show, in three independent instantiations, that acting on the relation damages relations nobody constrained.

### 7.1 Deleting the wrong edges splits nothing

The 15 confirmed different-person pairs are edges inside 7 identity groups. Removing exactly those 15 edges from the merge graph and recomputing connected components leaves **every group intact**. The reason is elementary and, in our experience, routinely forgotten: the two endpoints remain connected through other members of the group. Edge deletion is not a partition repair. We have replicated the effect three times across projects, which is why we treat it as a design rule: deleting an edge guarantees neither fewer errors nor fewer correct merges.

### 7.2 Constrained minimum colouring gives a lower bound, not a repair

The correct formulation of "these 15 pairs are not the same person, these 12 are" over a group's members is a constrained partition problem: contract must-link edges, then colour the conflict graph induced by must-differ edges, with the chromatic number as the minimum number of blocks. We solve it with DSATUR.

Result: **443 → 450 identities** (net +7). All 15 different-person pairs are separated, all 12 same-person pairs preserved, and T1's own metrics are untouched (P 1.000, R 0.147, 5 true / 0 false, identical to v0). A 15/15 and 12/12 result looks like success, and as a *constraint-satisfaction* statement it is: this is the correct thing for L3 to guarantee.

What it is not is a repair of the graph, and the distinction matters for the protocol. A 32-track group constrained by a *single* confirmed different-person pair can only shed one track: it splits into blocks of 31 and 1, and the other 30 tracks were never adjudicated. Net +7 identities on 443 means the operation addressed under 2% of the graph's structure. Constraint satisfaction therefore gives a **lower bound on the correction**, and L3's acceptance condition is written accordingly: "all adjudicated relations are honoured and no known-correct merge is broken", never "the graph is fixed".

### 7.3 A strong pairwise verifier damages the partition

A natural generalisation of 7.2 replaces sparse human labels with a dense automatic verifier: score every merged pair with an appearance signal and cut below a threshold. On the 27 adjudicated pairs the signals do discriminate, and this is the one place where appearance works well in D1:

**Table 8. Pairwise discrimination on the 27 contradiction pairs (12 same / 15 different).**

| Signal | AUC | 95% CI | Direction |
|---|---|---|---|
| OSNet per-camera mean-sub cosine | **0.833** | [0.639, 0.978] | same higher |
| E2 per-camera mean-sub cosine | 0.826 | n/a | same higher |
| Three-feature leave-one-out logistic | **0.856** | n/a | same higher |
| Box-height ratio | **0.233** | [upper 0.433] | **inverted** |

The last row is not incidental: for same-person pairs the height ratio is *lower*, so the same person's two boxes differ in scale more than two strangers' boxes do. This is consistent with the pose-class analysis of Section 6.1 and it explains why scale-based reasoning must be used to *select templates*, not to *reject pairs*.

Applied as a post-hoc splitter, cutting every merged pair below threshold and recomputing components, the verifier behaves as follows.

**Table 9. Partition-level effect of a pairwise verifier with AUC 0.833.**

| Configuration | Identities | T1 true pairs | T2 same-person preserved |
|---|---|---|---|
| v1 (human constraints only) | 450 | 5 | 12/12 |
| + verifier @ 0.20 | 475 | **4** | 10/12 |
| + verifier @ 0.25 | 483 | 4 | 10/12 |
| + verifier @ 0.30 | 491 | **2** | 9/12 |
| + verifier @ 0.40 | 511 | 1 | 7/12 |

At every operating point the verifier cuts all 15 confirmed errors and destroys correct merges it never addressed. At 0.30 it removes 3 of the 5 T1 true pairs. The mechanism is the same as 7.1 and 7.2: cutting an edge changes a transitive closure, so a *pairwise* decision has *partition-wide* consequences, and the pairs it cuts are not the pairs a human constrained.

The general statement is that *pairwise AUC 0.833 does not imply partition-level utility*. A signal good enough to rank review candidates is not thereby good enough to arbitrate a partition, so the protocol locates appearance signals in the ordering of the queue, where their discriminative power is an asset, and never in the decision.

### 7.4 Why this is structural, not a tuning failure

A better-calibrated verifier would not avoid the damage, for three reasons. (i) The damage is monotone in threshold (Table 9), so there is no operating point where the verifier is net-positive on T1. (ii) The damaged merges are *unconstrained*: no amount of labelling of the 27 pairs changes them, because the verifier acts on all pairs and the pairs that were right were never in the label set. (iii) Any pipeline that merges transitively has this property, so the argument applies to every agglomerative or graph-based identity builder, not to ours specifically.

The constructive corollary, and the reason L3 is specified the way it is: **budget human labels on partitions, not on pairs.** A pair label constrains a relation; what an operator needs constrained is a group. This inverts the usual active-learning instinct, and it is why the queue asks about *contradictions inside groups* rather than about pairs sampled from a similarity distribution.

A fourth instantiation belongs here because it concerns the clock rather than the graph. Adding the strictest temporal co-visibility gate, under which any simultaneous appearance across a camera set the survey calls disjoint blocks the merge, changes the identity count from 443 to 444, blocks exactly one merge and introduces one false merge.

**Table 10. Merge strategies at threshold 0.75 (D1, all with the attribute gate).**

| Strategy | Identities | Merges | T1 same | T1 diff | Merge precision |
|---|---|---|---|---|---|
| frame-index spans | 443 | 67 | 5/34 | 0/80 | 0.930 |
| OSD clock | 443 | 67 | **7/34** | 1/80 | 0.907 |
| OSD + strict co-visibility (>60 s) | 444 | 66 | 7/34 | 1/80 | 0.905 |
| OSD + strict co-visibility (>300 s) | 444 | 66 | 7/34 | 1/80 | 0.907 |

The clock correction is worth +2 true pairs at constant identity count and is a genuine measurement fix that any deployment should adopt. The hard gate is worth nothing, for the same reason as above: greedy merging has redundant paths, so blocking one edge does not prevent the same two clusters from joining through a third member. Geometry earns its place in the protocol as an *item selector* (L3) and as a *tier condition* (L1), not as an edge filter.

---

## 8 Deployment Cost and Operations

A protocol is only useful if its cost is stated, so this section reports what we actually spent and what an integrator repeats.

**Adjudication cost.** 27 items, 20–30 minutes, one adjudicator, on a phone, offline. Per item that is roughly 45–67 seconds, which is the number an integrator needs: it sets the queue length a shift can absorb (about 30–40 items per operator-hour). Sampling instants are chosen to maximise the minimum box height across the two sides rather than uniformly, which is what keeps the per-item time this low; uniform sampling lands on monitors and produces unreadable items that must be re-issued.

**Tooling cost.** The adjudication surface is a single self-contained HTML file with images embedded (2.05 MB for 27 items) that requires no network and no server, because the adjudicator is typically not at the workstation that generated the crops. Verdicts export as JSON and are ingested as constraints. Generating the file from a graph and a video set takes about 10 minutes on first run and is cached thereafter.

**Probe cost.** P1 and P3 run on existing detections and take under an hour each. P2 is a site task; for six cameras it is half a day including the walk-through. The whole diagnostic therefore fits in one working day, which is what makes "measure before you buy" a realistic recommendation.

**What does not scale for free.** Queue length grows linearly with capture duration and with the number of contradictions, and we have not measured the adjudication cost curve beyond 35 minutes. Two levers reduce it without changing the protocol: better item selection (rank by contradiction duration and by box size, both of which correlate with readability) and batching by group, since one group verdict constrains many pairs (Section 7.4).

---

## 9 Discussion

### 9.1 What is transferable

Four claims generalise beyond this site.

1. **The pixel budget.** A 60 px boundary separating AUC 0.352 from 0.847 is a property of human faces at surveillance distances, not of this building. Any site can measure its median face width in an afternoon, learn in advance whether face-dependent tiers are available, and read the remedy off the same number (median above 80 px).
2. **Regional co-visibility as the tier condition.** Every site with a glass wall, a corridor seen from both ends, or an open floor visible from multiple positions will find that pair-level co-visibility tables are wrong. The protocol's response, measuring co-visibility per region and buying L1 only where a region is shared, is directly transferable and cheaper than any model improvement.
3. **The structural argument of Section 7.** Transitive closure is universal, so the damage from local edits is universal. Pipelines that merge transitively should expect a pairwise verifier to harm them, and can test it in two lines: cut by verifier, recompute components, count known-correct merges.
4. **The reporting discipline of Section 4.3.** Version-matched comparison, stated denominators, and a refusal to let similarity certify purity are cheap and prevent the most common errors in this literature.

### 9.2 What is site-specific

- 443 identities, 56 cross-camera groups, 1,574 tracks and every AUC in Section 5 are properties of this capture, this detector, and this frozen encoder. The *sign* of each probe reading we expect to replicate; the *values* we do not.
- The attribute gate's 0.730 → 0.930 depends on the site's clothing diversity. A uniformed site shows the opposite failure: everyone agrees, and the gate becomes uninformative rather than wrong.
- The 27-item queue follows from 35 minutes and six cameras. Longer captures scale the queue linearly.

### 9.3 Design rules

Four rules follow, each the inverse of a common default.

- **Do not add a temporal hard gate to repair a merge rule.** It blocks one merge and changes nothing (Table 10). Spend the effort on a co-visibility *map* and use contradictions to select review items.
- **Do not put a verifier downstream of a partition.** Either put it upstream where it ranks candidates before any merge happens, or keep it out. Downstream, a pairwise cut substitutes damage for repair (Table 9).
- **Do put human labels on groups.** One group verdict constrains many pair relations; one pair verdict constrains one, and (Section 7.2) often fails to change the partition at all.
- **Do buy overlap before buying encoders.** The difference between the 0.904 pair and the 0.231 pair is geometric, and 5–10% shared view is a cabling decision, not a modelling one.

### 9.4 What the protocol changes at procurement

The practical consequence is a change in what gets specified. A tender that says "cross-camera ReID with ≥ 90% accuracy" is unanswerable and, in our data, unachievable. A tender that says "L0 everywhere; L1 for pairs certified co-visible by P2; L2 with attribute veto; L3 for 30 items per operator-hour; report merge precision on covered pairs together with the mixed-group fraction" is answerable, priceable, and verifiable after acceptance. That conversion, from a benchmark number to a set of measurable obligations, is the contribution we consider most likely to outlive the particular encoder used here.

---

## 10 Limitations

1. **Single site, single day.** One building, one 35-minute capture, one eight-segment corpus from the same floor. The co-visibility reversal and the probe boundaries are one site's geometry; the ordering of the probes is what we expect to hold.
2. **Single adjudicator, no inter-rater agreement.** 153 pairs were judged by one person. Internal consistency was verified (the same group never received contradictory verdicts across items) but Cohen's κ was not computed. This is the most serious limitation of the truth set and the first thing a replication should fix.
3. **T1 is not an unbiased sample.** It was drawn to test a rule, not to estimate a population rate. Every aggregate inherits that bias, which is why T1 and T2 are never pooled.
4. **No public-benchmark comparison under an identical protocol.** We verified that the same frozen stack behaves normally on MARS-style data (a pre-registered six-hypothesis check passed four), which supports the claim that the behaviour is site-induced rather than implementation-induced, but we did not re-run the partition-level experiments on a public corpus.
5. **Audit coverage is partial.** 15 of 56 cross-camera groups were touched by a contradiction; the other 41 produced no signal and remain unaudited. "No contradiction" is not "pure".
6. **L2+ evidence comes from D2.** The pose-conditioned template result is measured on 179 queries over 26 identities. The direction is stable across all 14 identities with sufficient samples, but the magnitude is conditional on the identity's pose span.
7. **Data cannot be released.** Privacy and site-authorisation constraints. We release protocol, parameters, truth-set structure, and adjudication rules; the crops shown are illustrative and non-identifying at publication scale.

---

## 11 Conclusion

We asked what an integrator can promise after installing six cameras, and answered with a measurement procedure. Three probes (face pixel budget, regional co-visibility, wardrobe diversity), each measurable in under a day and each gating a specific delivery tier, predict which identity functions a site can support. Applied here, they yield a ladder: 42.3% recall for same-camera cross-period identity (D2), AUC 0.904–0.994 inside the one genuinely co-visible region (D1), merge precision 0.730 → 0.930 from a semantic attribute veto at unchanged recall, selector recall 56.4% → 79.3% from pose-conditioned templates of which 11.8 points need no face, and a 27-item queue adjudicable in twenty to thirty minutes that honours 27 of 27 adjudicated constraints.

The ladder cannot be collapsed. Local constraints do not compose over a global partition: deleting wrong edges splits nothing, constrained colouring honours every label and moves the count by seven, and a verifier with pairwise AUC 0.833 destroys three of five correct merges. That result is why the protocol is tiered, and it is why appearance signals rank human attention rather than replace it.

Two consequences for practice. The binding constraint is capture geometry, not model choice: a median face width above 80 px, or 5–10% view overlap between adjacent cameras, changes which tiers exist, and both are decided at cabling time. And the deliverable that survives acceptance is a pair of numbers reported together: precision on covered pairs, and the fraction of audited groups that turn out mixed.

---

## CRediT authorship contribution statement

**[Author 1]:** Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Visualization, Writing – original draft. **[Author 2]:** Methodology, Validation, Writing – review & editing, Supervision. **[Author 3]:** Resources, Project administration, Funding acquisition, Writing – review & editing.

*All authors approved the final version of the manuscript and agree to be accountable for the work.*

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Data availability

The raw video, track tables and crops record identifiable individuals on a client site under an authorisation that prohibits redistribution, so they cannot be released. Released with this article: the protocol specification, the probe thresholds, the structure and sampling mechanism of both truth sets, the adjudication rules, every result file cited in Appendix A, and the scripts that generate them. Researchers who need the underlying media for independent verification may request access under a data-use agreement from the corresponding author.

## Declaration of generative AI and AI-assisted technologies

During the preparation of this work the authors used a large language model to draft and edit prose and to check internal consistency of reported figures. After using this tool the authors reviewed and edited the content and take full responsibility for the content of the publication.

## Acknowledgements

The authors thank [site contact] for facilitating capture on the office floor, and [reviewer/colleague] for comments on an earlier draft.

---

## References

[1] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, Scalable person re-identification: a benchmark, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1116–1124.

[2] L. Wei, S. Zhang, W. Gao, Q. Tian, Person transfer GAN to bridge domain gap for person re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 79–88.

[3] K. Zhou, Y. Yang, A. Cavallaro, T. Xiang, Omni-scale feature learning for person re-identification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 3702–3712.

[4] S. He, H. Luo, P. Wang, F. Wang, H. Li, W. Jiang, TransReID: transformer-based object re-identification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[5] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S.C.H. Hoi, Deep learning for person re-identification: a survey and outlook, IEEE Trans. Pattern Anal. Mach. Intell. 44 (2021) 2872–2893.

[6] N. Tang, Z. Fan, Y. Zhu, T. Zhang, Unsupervised person re-identification via camera-aware multi-level label refinement, Neural Netw. 195 (2025) 108292.

[7] J. Wang, H. Fu, L. Dou, CARR: camera-aware reciprocal refinement for unsupervised visible-infrared person re-identification, Appl. Soft Comput. 188 (2026) 114445.

[8] Y. Xiao, T. Zhu, F. Guan, H. Sun, J. Huang, A. Grau Saldes, Cross-camera reliability modeling with soft contrastive learning for unsupervised person re-identification, Knowl.-Based Syst. (2026), in press.

[9] Y. Ge, F. Zhu, D. Chen, R. Zhao, H. Li, Self-paced contrastive learning with hybrid memory for domain adaptive object re-ID, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 33, 2020.

[10] E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi, Performance measures and a data set for multi-target, multi-camera tracking, in: European Conference on Computer Vision Workshops (ECCVW), 2016, pp. 17–35.

[11] E. Ristani, C. Tomasi, Features for multi-target multi-camera tracking and re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6036–6046.

[12] AI City Challenge, Track 1: multi-camera 3D perception (Sim2Real), 2026.

[13] Y. Geifman, R. El-Yaniv, Selective classification for deep neural networks, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, 2017.

[14] A.N. Angelopoulos, S. Bates, E.J. Candès, M.I. Jordan, L. Lei, Learn then test: calibrating predictive algorithms to achieve risk control, Ann. Appl. Stat. 19 (2025).

[15] J. Wang, J. Cao, C. Song, Gallery-distance profiles for open-set person re-identification with frozen encoders in camera networks, Sensors 26 (2026) 5479.

[16] Y. Ding, X. Chen, Y. Zhang, X. Huang, Smart building evacuation by tracking multi-camera network and explainable re-identification model, Eng. Appl. Artif. Intell. 148 (2025) 110394.

[17] K.L. Wagstaff, Machine learning that matters, in: Proceedings of the International Conference on Machine Learning (ICML), 2012.

[18] D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison, Hidden technical debt in machine learning systems, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 28, 2015.

[19] A. Paleyes, R.-G. Urma, N.D. Lawrence, Challenges in deploying machine learning: a survey of case studies, ACM Comput. Surv. 55 (2022).

[20] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang, ByteTrack: multi-object tracking by associating every detection box, in: European Conference on Computer Vision (ECCV), 2022, pp. 1–21.

[21] Y. Du, Z. Zhao, Y. Song, Y. Zhao, F. Su, T. Gong, H. Meng, StrongSORT: make DeepSORT great again, IEEE Trans. Multimedia 25 (2023).

[22] J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, HOTA: a higher order metric for evaluating multi-object tracking, Int. J. Comput. Vis. 129 (2021) 548–578.

[23] GB/T 31488-2015, Technical Requirements for Video Surveillance Face Recognition Systems, Standardization Administration of China, Beijing, 2015.

---

## Appendix A. Reproducibility

All experiments are deterministic given the track tables and embeddings; no training is involved. Entry points are listed below (repository paths); `EVIDENCE.md` maps every number in this paper to its source file and command.

**Diagnostic probes**
- Cross-camera AUC sweep (2 representations × 3 alignments): `tmp/full/emb_auc.py` → `emb_auc_out.txt`
- Per-pair stratification: same file, `--by-pair`
- Counterfactual gate (co-visibility map, attribute veto): `tmp/full/attr_filter_ingraph.py`

**Tiered protocol**
- Identity graph construction and merge-strategy comparison: `tmp/full/spacetime_ingraph.py` (`MODES=base,osd,covis60,covis300 THR=0.75`)
- Temporal audit on the OSD clock: `tmp/full/spacetime_audit.py`
- Review-queue generation: `tools/build_spacetime_review.py`
- Adjudication ingestion: `tools/ingest_spacetime_review.py`
- Constrained minimum colouring: `tools/apply_spacetime_truth.py`
- Pose-conditioned templates and pool: `tmp/full/exp_pose_pool.py`, `tmp/full/pose_pr.py`
- Face-anchor pool: `tmp/full/exp_anchor_pool.py`
- Runtime selector: `tmp/full/exp_selector_runtime.py`

**Non-composability experiments**
- Pairwise verifier AUC and threshold sweep: `tmp/full/spacetime_verifier.py`
- Partition-level splitter: `tmp/full/verifier_split.py`

**Machine-readable results**: `spacetime_audit.json`, `spacetime_ingraph.json`, `spacetime_signal.json`, `verifier_split.json`, `long35_crosscam_attr8.json`, `long35_crosscam_attr8_v1.json`, `emb_auc_out.txt`, `verifier_out.txt`.

The raw video, track tables, and crops are not releasable (privacy and site authorisation). Reviewers requiring access for verification may request it under an agreement.

---

## Appendix B. Truth-set composition

**Table B1. Adjudicated pairs by source and verdict.**

| Source | Pairs | Same | Different | Camera pairs | Purpose |
|---|---|---|---|---|---|
| T1 (prior-sampled) | 126 | 34 | 92 | 7 | test the merge rule; measures recall preservation |
| T2 (contradiction-triggered) | 27 | 12 | 15 | 10 | test constraint repair; measures precision at the failure surface |
| **Total** | **153** | **46** | **107** | **12** (union) | n/a |

**Table B2. T2 group-level outcome.**

| Measure | Value |
|---|---|
| Contradictions audited | 27 |
| Identity groups touched | 15 |
| Groups mixed after adjudication | 7 |
| Groups pure after adjudication | 8 |
| Pairs outside T1 coverage | 20 |
| Pairs partially covered by T1 | 6 |
| Longest same-person co-occurrence across a "disjoint" pair | 815 s (305:173 vs 320:45) |

Adjudication artefacts: `deliverable/spacetime_review/{items.json, labels_20261112.json, gt_spacetime_items.json, ingest_report.md, apply_v1_report.md}`.

