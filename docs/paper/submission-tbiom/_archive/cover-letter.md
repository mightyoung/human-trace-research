# Cover letter — IEEE Transactions on Biometrics, Behavior, and Identity Science

[Date]

Dear Professor Nixon,

Please consider the enclosed manuscript, "How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification", for publication in *IEEE Transactions on Biometrics, Behavior, and Identity Science* as a Regular Paper.

**Scope.** The manuscript is an evaluation study, and we submit it as one. T-BIOM's call for papers lists, among its broad topics, "novel (challenging) biometric databases, protocol and benchmarking, large scale evaluation, confidence interval estimation, and performance modeling and prediction", together with "person re-identification" and "detection, tracking, identification, and de-identification in surveillance". The manuscript is a contribution to that first item, made on a surveillance identity problem, and its central artefact is a protocol: three label-free site probes that decide which identity functions a camera network can support, and five delivery tiers each carrying an acceptance condition, a measured benefit and a stated failure mode.

**The problem.** Re-identification models report above 90% mAP on benchmarks. An integrator who has just installed six cameras in an office cannot act on that number, because the benchmark's conditions (tight upright crops, viewpoints shared with training, no seated subjects at 50 px) are not the conditions of a building. The question that decides the project is which identity functions the network can support, and whether that can be established before the contract is signed.

**What the manuscript reports.** We freeze an off-the-shelf OSNet and never tune it on the site, then measure where a learned re-identification representation stops being informative. Two captures of one office floor (six cameras over 35 minutes; an eight-segment corpus with 26 named identities) yield 1,574 tracks, 443 hypothesised identities and 153 adjudicated track pairs. Every headline number is reported with a population and an interval: the gallery-membership truth layer was re-annotated blind by a second person (257 tracks, Cohen's κ = 0.925, 95% CI 0.891–0.958); the pair layer was judged by one person and this is stated as the study's most serious limitation rather than left implicit. The measured outcomes are: 42.3% recall for same-camera identity across time; AUC 0.904 and 0.994 inside the one genuinely co-visible camera pair; merge precision from 0.730 to 0.930 under an attribute veto at unchanged recall; selector recall from 56.4% to 79.3% with pose-conditioned templates, of which 11.8 points require no face; and a 27-item adjudication queue that resolves in 20 to 30 minutes and satisfies every adjudicated constraint. Outside that one co-visible pair, six encoder and alignment configurations all sit at chance (AUC 0.450 to 0.622, every interval covering 0.5).

**The structural result.** The tiers cannot be collapsed into a single automatic stage, and we give three independent demonstrations that a local constraint on a global partition damages correct merges it never addressed: deleting the adjudicated-wrong edges splits zero groups, constrained minimum colouring honours all 27 labels but moves the identity count from 443 to 450, and a verifier with pairwise AUC 0.833 destroys three of five known-correct merges monotonically in its threshold. This is why appearance signals belong in the ordering of human review rather than in the decision, and it is the reason the paper is a protocol rather than a method.

**Declarations.** The manuscript is original, has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. The raw video cannot be released because it records identifiable individuals on a client site under an authorisation that forbids redistribution; the protocol, probe thresholds, truth-set structure, adjudication rules, all result files and the scripts that generate them are released with the article, and we will supply the media to reviewers under a data-use agreement if the editors require independent verification.

Suggested reviewers can be supplied on request.

Yours sincerely,

[Name], on behalf of all authors
[Department, Institution]
[E-mail]
