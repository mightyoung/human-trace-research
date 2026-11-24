# Cover letter

[Date]

Dear Editors,

Please consider the enclosed manuscript, "How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification", for publication in *Neurocomputing* as a Full Length Article.

**The problem.** Cross-camera re-identification models report above 90% mAP on benchmarks. An integrator who has just installed six cameras in an office cannot use that number, because the benchmark's conditions (tight upright crops, viewpoints shared with training, no seated subjects at 50 px) are not the conditions of a building. The question that decides the project is which identity functions the network can support, and whether that can be established before the contract is signed.

**What the manuscript does.** We freeze an off-the-shelf OSNet and do not tune it on the site, then measure where a learned re-identification representation stops being informative. Two captures of one office floor (six cameras over 35 minutes; an eight-segment corpus with 26 named identities) yield 1,574 tracks, 443 hypothesised identities and 153 adjudicated track pairs. The availability question reduces to three label-free site probes, each gating one delivery tier; three independent experiments then show that the tiers cannot be replaced by a single automatic stage.

**Why we believe it belongs in *Neurocomputing*.** The paper studies the operating envelope of a learned representation: which conditions preserve its discriminative power, which destroy it, and what a system should do at the boundary. The three probes are a measurement procedure for that envelope and the tiered protocol is the decision rule that follows. The central structural result, that a pairwise signal with AUC 0.833 can still damage a partition it was never asked about, concerns how learned similarity interacts with transitive clustering, and it applies to any agglomerative identity builder rather than to our pipeline alone.

The measured outcomes are concrete enough to be checked: 42.3% recall for same-camera cross-period identity; AUC 0.904 and 0.994 inside the one genuinely co-visible camera pair; merge precision moving from 0.730 to 0.930 under an attribute veto at unchanged recall; selector recall from 56.4% to 79.3% with pose-conditioned templates, of which 11.8 points require no face; and a 27-item adjudication queue that resolves in 20 to 30 minutes and satisfies every adjudicated constraint. Outside that one co-visible pair, six encoder and alignment configurations all sit at chance, which is why the protocol falls back to geometry, semantics and operator review instead of promising automatic identity.

**Declarations.** The manuscript is original, has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. The raw video cannot be released because it records identifiable individuals on a client site under an authorisation that forbids redistribution; the protocol, probe thresholds, truth-set structure, adjudication rules, all result files and the scripts that generate them are released with the article, and we will supply the media to reviewers under a data-use agreement if the editors require independent verification.

Suggested reviewers can be supplied on request.

Yours sincerely,

[Name], on behalf of all authors
[Department, Institution]
[E-mail]
