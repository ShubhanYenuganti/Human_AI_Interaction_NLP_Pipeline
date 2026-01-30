-- =============================================================================
-- Creates a table of the counts of extracted evidence by excerpt type followed by a json object of subcategories and their counts
-- =============================================================================
CREATE TABLE public.counts (
    excerpt_type text NOT NULL PRIMARY KEY,
    total_count integer NOT NULL,
    category_counts jsonb DEFAULT '{}'::jsonb
);

INSERT INTO public.counts (
    excerpt_type,
    total_count,
    category_counts
)
SELECT
    excerpt_type,
    COUNT(*)::integer AS total_count,
    COALESCE(
        (
            SELECT jsonb_object_agg(first_key_value, cnt)
            FROM (
                SELECT e2.metadata->>fk.key AS first_key_value, COUNT(*)::integer AS cnt
                FROM public.extracted_evidence e2
                CROSS JOIN LATERAL (
                    SELECT key FROM jsonb_object_keys(e2.metadata) AS k(key) ORDER BY 1 LIMIT 1
                ) fk
                WHERE e2.excerpt_type = e1.excerpt_type
                  AND e2.metadata IS NOT NULL
                  AND jsonb_typeof(e2.metadata) = 'object'
                  AND e2.metadata <> '{}'::jsonb
                GROUP BY e2.metadata->>fk.key
            ) sub
        ),
        '{}'::jsonb
    ) AS category_counts
FROM public.extracted_evidence e1
GROUP BY excerpt_type;