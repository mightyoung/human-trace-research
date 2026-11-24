-- paper_tbiom 适配器投影：把通用 claim 还原成 T-BIOM 论文台账的老列名。
-- 只给 ops_claims.py / 老报告读；写入一律写 claim + claim_anchor，不写本视图。
-- 项目私有字段（section / evidence_line / label / domain / 原 kind）存在 claim.ext_json，由本视图展开。

DROP VIEW IF EXISTS paper_claim;

CREATE VIEW paper_claim AS
SELECT
  c.id                                            AS id,
  c.locator                                       AS display_id,
  json_extract(c.ext_json, '$.section')           AS section,
  json_extract(c.ext_json, '$.evidence_line')     AS evidence_line,
  json_extract(c.ext_json, '$.label')             AS label,
  c.value_text                                    AS value_text,
  c.stratum                                       AS domain,
  COALESCE(json_extract(c.ext_json, '$.paper_kind'), c.kind) AS kind,
  c.evidence_grade                                AS source_level,
  c.provenance_json                               AS provenance_json,
  c.verdict                                       AS verdict,
  c.reason                                        AS reason,
  c.recomputed_text                               AS recomputed_text,
  c.eval_run_id                                   AS eval_run_id,
  NULLIF((SELECT json_group_array(json(ref_json))
            FROM claim_anchor a
           WHERE a.claim_id = c.id AND a.type = 'tex_table'), '[]') AS tex_anchor_json,
  c.created_at                                    AS created_at,
  c.updated_at                                    AS updated_at
FROM claim c
WHERE c.study_id = 'st_multicam-reid';
