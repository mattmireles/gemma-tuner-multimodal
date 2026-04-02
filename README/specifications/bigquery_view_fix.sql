-- Fixed BigQuery VIEW: verified_generations_with_audio
-- This view adds a synthetic numeric ID column to avoid type inference issues
-- while preserving all metadata fields including note_id as a STRING

CREATE OR REPLACE VIEW `YOUR_PROJECT_ID.whisper_finetuning.verified_generations_with_audio` AS
WITH unioned AS (
  -- truthgen_results_* snapshots
  SELECT
    COALESCE(
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_path'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_uri'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_url')
    ) AS audio_path,
    JSON_VALUE(TO_JSON_STRING(t), '$.ground_truth_transcript')      AS ground_truth_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.verbatim_transcript')          AS verbatim_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.what_im_really_thinking')      AS what_im_really_thinking,
    JSON_VALUE(TO_JSON_STRING(t), '$.clean_rewrite')                AS clean_rewrite,
    CAST(JSON_VALUE(TO_JSON_STRING(t), '$.audio_duration') AS FLOAT64) AS audio_duration,
    JSON_VALUE(TO_JSON_STRING(t), '$.record_id')                    AS record_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.chunk_id')                     AS chunk_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.note_id')                      AS note_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.user_id')                      AS user_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.timestamp')                    AS timestamp,
    CONCAT('truthgen_results_', _TABLE_SUFFIX)                      AS source_table
  FROM `YOUR_PROJECT_ID.whisper_finetuning.truthgen_results_*` AS t

  UNION ALL

  -- Talktastic (tt-10, tt-test, tt-test_*)
  SELECT
    COALESCE(
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_path'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_uri'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_url')
    ) AS audio_path,
    JSON_VALUE(TO_JSON_STRING(t), '$.ground_truth_transcript')      AS ground_truth_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.verbatim_transcript')          AS verbatim_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.what_im_really_thinking')      AS what_im_really_thinking,
    JSON_VALUE(TO_JSON_STRING(t), '$.clean_rewrite')                AS clean_rewrite,
    CAST(JSON_VALUE(TO_JSON_STRING(t), '$.audio_duration') AS FLOAT64) AS audio_duration,
    JSON_VALUE(TO_JSON_STRING(t), '$.record_id')                    AS record_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.chunk_id')                     AS chunk_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.note_id')                      AS note_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.user_id')                      AS user_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.timestamp')                    AS timestamp,
    CONCAT('tt-', _TABLE_SUFFIX)                                    AS source_table
  FROM `YOUR_PROJECT_ID.whisper_finetuning.tt-*` AS t

  UNION ALL

  -- Oasis (all tables starting with oasis)
  SELECT
    COALESCE(
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_path'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_uri'),
      JSON_VALUE(TO_JSON_STRING(t), '$.audio_url')
    ) AS audio_path,
    JSON_VALUE(TO_JSON_STRING(t), '$.ground_truth_transcript')      AS ground_truth_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.verbatim_transcript')          AS verbatim_transcript,
    JSON_VALUE(TO_JSON_STRING(t), '$.what_im_really_thinking')      AS what_im_really_thinking,
    JSON_VALUE(TO_JSON_STRING(t), '$.clean_rewrite')                AS clean_rewrite,
    CAST(JSON_VALUE(TO_JSON_STRING(t), '$.audio_duration') AS FLOAT64) AS audio_duration,
    JSON_VALUE(TO_JSON_STRING(t), '$.record_id')                    AS record_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.chunk_id')                     AS chunk_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.note_id')                      AS note_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.user_id')                      AS user_id,
    JSON_VALUE(TO_JSON_STRING(t), '$.timestamp')                    AS timestamp,
    CONCAT('oasis', _TABLE_SUFFIX)                                  AS source_table
  FROM `YOUR_PROJECT_ID.whisper_finetuning.oasis*` AS t
)
SELECT
  -- Add synthetic numeric ID as first column for compatibility with export tools
  ROW_NUMBER() OVER (
    ORDER BY record_id, chunk_id, audio_path
  ) AS id,
  
  -- Original columns
  audio_path,
  ground_truth_transcript,
  verbatim_transcript,
  what_im_really_thinking,
  clean_rewrite,
  
  -- Best-available text (you can still pick any field at query time)
  COALESCE(
    NULLIF(ground_truth_transcript, ''),
    NULLIF(clean_rewrite, ''),
    NULLIF(verbatim_transcript, ''),
    NULLIF(what_im_really_thinking, '')
  ) AS best_available_text,
  
  audio_duration,
  record_id,
  chunk_id,
  note_id,  -- Keep as STRING metadata field
  user_id,
  timestamp,
  source_table
FROM (
  SELECT
    u.*,
    ROW_NUMBER() OVER (
      PARTITION BY record_id, chunk_id, audio_path
      ORDER BY source_table DESC
    ) AS rn
  FROM unioned AS u
  WHERE
    ground_truth_transcript IS NOT NULL
    AND TRIM(ground_truth_transcript) != ''
    AND audio_path IS NOT NULL
    AND TRIM(audio_path) != ''
    AND REGEXP_CONTAINS(LOWER(audio_path), r'\.(wav|mp3|m4a|aac|ogg|flac|webm|opus|aiff|aif|caf)(?:\?|$)')
)
WHERE rn = 1
  AND (audio_duration IS NULL OR audio_duration > 0);