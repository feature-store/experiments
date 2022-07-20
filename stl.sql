-- ## Workload
-- The source table that's being constantly appended to
CREATE TABLE cpu_readings AS (
    id INT,
    reading_timestamp FLOAT,
    avg_cpu FLOAT
)
INSERT INTO cpu_readings VALUES (101, 1648856926.123, 4.2)

-- The feature table derived from source table
CREATE TABLE cpu_forecast AS (
    id INT PRIMARY KEY UNIQUE,
    y_train_last_timestamp FLOAT, -- used to compute offset indexing into predicted_avg_cpu_array
    y_pred_future_readings TEXT, -- use array type if supported natively, this can also be flattened.
)

-- Periodically the system query for prediction, say every 5min.
SELECT id, extract_from_forecast(y_train_last_timestamp, now(), y_pred_future_readings) AS predicted_cpu
  FROM cpu_forecast
 WHERE id = ?
--


-- ## Conceptually, this can is a materialized view
CREATE MATERIALIZED VIEW cpu_forecast AS (
    SELECT id,
           max(reading_timestamp) AS y_train_last_timestamp,
           run_stl_forecast(avg_cpu ORDER BY reading_timestamp) AS y_pred_future_readings,
    FROM cpu_readings
    WHERE reading_timestamp > now() - window_size
    GROUP BY id
)
-- In postgres we can automate the refresh with a trigger.
CREATE OR REPLACE FUNCTION refresh_view()
RETURNS TRIGGER LANGUAGE plpgsql
AS $$
BEGIN
  REFRESH MATERIALIZED VIEW cpu_forecast;
  RETURN NULL;
END $$;

CREATE TRIGGER refresh_post_search
AFTER INSERT
ON cpu_readings
FOR EACH STATEMENT
EXECUTE PROCEDURE refresh_post_search();
--


-- ## RALF Experiments
-- But this is too expensive and inefficient, `run_stl_forecast` is very expensive to run.
-- Therefore RALF will orchestrate the following upsert query based on its policies.
INSERT INTO cpu_forecast(id, y_train_last_timestamp, y_pred_future_readings) VALUES
(
    SELECT id,
        max(reading_timestamp) AS y_train_last_timestamp,
        run_stl_forecast(avg_cpu ORDER BY reading_timestamp) AS y_pred_future_readings,
    FROM cpu_readings
    WHERE reading_timestamp > now() - window_size
      AND id = ?
    GROUP BY id
)
ON CONFLICT(id) DO UPDATE SET *
