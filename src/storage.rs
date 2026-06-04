use crate::portfolio::{InterimMetadata, TrialResult};
use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::format::KeyValue;
use std::fs::File;
use std::sync::Arc;

pub struct ParquetStorage {
    schema: SchemaRef,
    writer: Option<ArrowWriter<File>>,
    batch_buffer: Vec<DefaultRecord>,
    batch_size: usize,
    output_path: String,
    key_value_metadata: Vec<KeyValue>,
}

/// One Parquet row: a single defaulted obligor in a single trial. Non-defaulted
/// asset-trials are not written (the store is sparse over defaults), so totals
/// the analyzer needs — trial/asset counts, default rates — live in the
/// file-level key-value metadata instead.
#[derive(Debug, Clone)]
pub struct DefaultRecord {
    pub trial_id: i64,
    pub asset_id: i32,
    pub sector_id: i32,
    pub sector_name: String,
    pub default_period: i32,
    pub time_to_default: f32,
    pub loss_amount: f32,
    pub recovery_rate: f32,
    pub asset_value: f32,
    pub systematic_factor: f32,
    pub idiosyncratic_factor: f32,
    pub pd: f32,
    pub lgd_mean: f32,
    pub exposure: f32,
}

impl ParquetStorage {
    pub fn new(output_path: &str, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(ParquetStorage {
            schema: create_parquet_schema(),
            writer: None,
            batch_buffer: Vec::with_capacity(batch_size),
            batch_size,
            output_path: output_path.to_string(),
            key_value_metadata: Vec::new(),
        })
    }

    pub fn set_metadata(&mut self, meta: &InterimMetadata) {
        self.key_value_metadata = vec![
            KeyValue {
                key: "simflux.n_trials".to_string(),
                value: Some(meta.n_trials.to_string()),
            },
            KeyValue {
                key: "simflux.n_assets".to_string(),
                value: Some(meta.n_assets.to_string()),
            },
            KeyValue {
                key: "simflux.sector_asset_counts".to_string(),
                value: Some(sector_counts_json(
                    &meta.sector_names,
                    &meta.sector_asset_counts,
                )),
            },
        ];
    }

    pub fn initialize_writer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(&self.output_path)?;

        let mut props =
            WriterProperties::builder().set_compression(parquet::basic::Compression::SNAPPY);
        if !self.key_value_metadata.is_empty() {
            props = props.set_key_value_metadata(Some(self.key_value_metadata.clone()));
        }

        let writer = ArrowWriter::try_new(file, self.schema.clone(), Some(props.build()))?;
        self.writer = Some(writer);

        Ok(())
    }

    pub fn add_trial_results(
        &mut self,
        trial_results: &[TrialResult],
        sector_names: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for trial in trial_results {
            for ev in &trial.default_events {
                let record = DefaultRecord {
                    trial_id: trial.trial_id as i64,
                    asset_id: ev.asset_id as i32,
                    sector_id: ev.sector_id as i32,
                    sector_name: sector_names
                        .get(ev.sector_id as usize)
                        .cloned()
                        .unwrap_or_else(|| "Unknown".to_string()),
                    default_period: ev.default_period as i32,
                    time_to_default: ev.time_to_default as f32,
                    loss_amount: ev.loss_amount as f32,
                    recovery_rate: ev.recovery_rate as f32,
                    asset_value: ev.asset_value as f32,
                    systematic_factor: ev.systematic_factor as f32,
                    idiosyncratic_factor: ev.idiosyncratic_factor as f32,
                    pd: ev.pd as f32,
                    lgd_mean: ev.lgd_mean as f32,
                    exposure: ev.exposure as f32,
                };

                self.batch_buffer.push(record);

                if self.batch_buffer.len() >= self.batch_size {
                    self.write_batch()?;
                }
            }
        }

        Ok(())
    }

    pub fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // A run with zero defaults still must produce a readable, schema-bearing
        // file (the analyzer reads metadata from it), so initialize even when the
        // buffer is empty.
        if !self.batch_buffer.is_empty() {
            self.write_batch()?;
        } else if self.writer.is_none() {
            self.initialize_writer()?;
        }

        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }

        Ok(())
    }

    fn write_batch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.batch_buffer.is_empty() {
            return Ok(());
        }

        let batch = self.create_record_batch()?;

        if self.writer.is_none() {
            self.initialize_writer()?;
        }
        if let Some(writer) = &mut self.writer {
            writer.write(&batch)?;
        }

        self.batch_buffer.clear();
        Ok(())
    }

    fn create_record_batch(&self) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        let trial_ids = Int64Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.trial_id)
                .collect::<Vec<_>>(),
        );
        let asset_ids = Int32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.asset_id)
                .collect::<Vec<_>>(),
        );
        let sector_ids = Int32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.sector_id)
                .collect::<Vec<_>>(),
        );
        let sector_names = StringArray::from(
            self.batch_buffer
                .iter()
                .map(|r| r.sector_name.as_str())
                .collect::<Vec<_>>(),
        );
        let default_periods = Int32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.default_period)
                .collect::<Vec<_>>(),
        );
        let time_to_default = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.time_to_default)
                .collect::<Vec<_>>(),
        );
        let loss_amounts = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.loss_amount)
                .collect::<Vec<_>>(),
        );
        let recovery_rates = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.recovery_rate)
                .collect::<Vec<_>>(),
        );
        let asset_values = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.asset_value)
                .collect::<Vec<_>>(),
        );
        let sys_factor = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.systematic_factor)
                .collect::<Vec<_>>(),
        );
        let idio_factor = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.idiosyncratic_factor)
                .collect::<Vec<_>>(),
        );
        let pds = Float32Array::from(self.batch_buffer.iter().map(|r| r.pd).collect::<Vec<_>>());
        let lgd_means = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.lgd_mean)
                .collect::<Vec<_>>(),
        );
        let exposures = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.exposure)
                .collect::<Vec<_>>(),
        );

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(trial_ids),
                Arc::new(asset_ids),
                Arc::new(sector_ids),
                Arc::new(sector_names),
                Arc::new(default_periods),
                Arc::new(time_to_default),
                Arc::new(loss_amounts),
                Arc::new(recovery_rates),
                Arc::new(asset_values),
                Arc::new(sys_factor),
                Arc::new(idio_factor),
                Arc::new(pds),
                Arc::new(lgd_means),
                Arc::new(exposures),
            ],
        )?;

        Ok(batch)
    }
}

fn create_parquet_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("trial_id", DataType::Int64, false),
        Field::new("asset_id", DataType::Int32, false),
        Field::new("sector_id", DataType::Int32, false),
        Field::new("sector", DataType::Utf8, false),
        Field::new("default_period", DataType::Int32, false),
        Field::new("time_to_default", DataType::Float32, false),
        Field::new("loss_amount", DataType::Float32, false),
        Field::new("recovery_rate", DataType::Float32, false),
        Field::new("asset_value", DataType::Float32, false),
        Field::new("systematic_factor", DataType::Float32, false),
        Field::new("idiosyncratic_factor", DataType::Float32, false),
        Field::new("pd", DataType::Float32, false),
        Field::new("lgd_mean", DataType::Float32, false),
        Field::new("exposure", DataType::Float32, false),
    ]))
}

/// Build a minimal JSON object `{"<sector>": <count>, ...}` for the metadata
/// value (avoids a serde_json dependency). Sector names are escaped for `"`/`\`.
fn sector_counts_json(names: &[String], counts: &[usize]) -> String {
    let mut s = String::from("{");
    for (i, (name, count)) in names.iter().zip(counts.iter()).enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        for ch in name.chars() {
            match ch {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                _ => s.push(ch),
            }
        }
        s.push_str("\":");
        s.push_str(&count.to_string());
    }
    s.push('}');
    s
}

pub fn write_simulation_results_to_parquet(
    trial_results: &[TrialResult],
    output_path: &str,
    sector_names: &[String],
    batch_size: Option<usize>,
    meta: &InterimMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = ParquetStorage::new(output_path, batch_size.unwrap_or(10000))?;
    storage.set_metadata(meta);
    storage.add_trial_results(trial_results, sector_names)?;
    storage.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_record_batch() {
        let mut storage = ParquetStorage::new("test.parquet", 1000).unwrap();

        storage.batch_buffer.push(DefaultRecord {
            trial_id: 0,
            asset_id: 1,
            sector_id: 0,
            sector_name: "Tech".to_string(),
            default_period: 0,
            time_to_default: 0.5,
            loss_amount: 1000.0,
            recovery_rate: 0.4,
            asset_value: -2.0,
            systematic_factor: -1.5,
            idiosyncratic_factor: -0.8,
            pd: 0.05,
            lgd_mean: 0.6,
            exposure: 10000.0,
        });

        let batch = storage.create_record_batch();
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 14);
    }

    #[test]
    fn test_sector_counts_json() {
        let names = vec!["Tech".to_string(), "Finance".to_string()];
        let counts = vec![5usize, 3usize];
        assert_eq!(
            sector_counts_json(&names, &counts),
            "{\"Tech\":5,\"Finance\":3}"
        );
    }
}
