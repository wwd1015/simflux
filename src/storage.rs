use crate::portfolio::TrialResult;
use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;

pub struct ParquetStorage {
    schema: SchemaRef,
    writer: Option<ArrowWriter<File>>,
    batch_buffer: Vec<AssetRecord>,
    batch_size: usize,
    output_path: String,
}

#[derive(Debug, Clone)]
pub struct AssetRecord {
    pub trial_id: i64,
    pub asset_id: i32,
    pub sector_id: i32,
    pub sector_name: String,
    pub defaulted: bool,
    pub time_to_default: Option<f32>,
    pub loss_amount: f32,
    pub recovery_rate: f32,
    pub asset_value: f32,
    pub systematic_factor_global: f32,
    pub systematic_factor_sector: f32,
    pub pd: f32,
    pub lgd_mean: f32,
    pub exposure: f32,
}

impl ParquetStorage {
    pub fn new(output_path: &str, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = create_parquet_schema();

        Ok(ParquetStorage {
            schema,
            writer: None,
            batch_buffer: Vec::with_capacity(batch_size),
            batch_size,
            output_path: output_path.to_string(),
        })
    }

    pub fn initialize_writer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(&self.output_path)?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(file, self.schema.clone(), Some(props))?;
        self.writer = Some(writer);

        Ok(())
    }

    pub fn add_trial_results(
        &mut self,
        trial_results: &[TrialResult],
        sector_names: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for trial in trial_results {
            for asset_result in &trial.asset_results {
                let record = AssetRecord {
                    trial_id: trial.trial_id as i64,
                    asset_id: asset_result.asset_id as i32,
                    sector_id: asset_result.sector_id as i32,
                    sector_name: sector_names
                        .get(asset_result.sector_id as usize)
                        .unwrap_or(&"Unknown".to_string())
                        .clone(),
                    defaulted: asset_result.defaulted,
                    time_to_default: asset_result.time_to_default.map(|t| t as f32),
                    loss_amount: asset_result.loss_amount as f32,
                    recovery_rate: asset_result.recovery_rate as f32,
                    asset_value: asset_result.asset_value as f32,
                    systematic_factor_global: asset_result.systematic_factor_global as f32,
                    systematic_factor_sector: asset_result.systematic_factor_sector as f32,
                    pd: asset_result.pd as f32,
                    lgd_mean: asset_result.lgd_mean as f32,
                    exposure: asset_result.exposure as f32,
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
        if !self.batch_buffer.is_empty() {
            self.write_batch()?;
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

        let defaulted = BooleanArray::from(
            self.batch_buffer
                .iter()
                .map(|r| r.defaulted)
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

        let sys_factor_global = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.systematic_factor_global)
                .collect::<Vec<_>>(),
        );

        let sys_factor_sector = Float32Array::from(
            self.batch_buffer
                .iter()
                .map(|r| r.systematic_factor_sector)
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
                Arc::new(defaulted),
                Arc::new(time_to_default),
                Arc::new(loss_amounts),
                Arc::new(recovery_rates),
                Arc::new(asset_values),
                Arc::new(sys_factor_global),
                Arc::new(sys_factor_sector),
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
        Field::new("defaulted", DataType::Boolean, false),
        Field::new("time_to_default", DataType::Float32, true),
        Field::new("loss_amount", DataType::Float32, false),
        Field::new("recovery_rate", DataType::Float32, false),
        Field::new("asset_value", DataType::Float32, false),
        Field::new("systematic_factor_global", DataType::Float32, false),
        Field::new("systematic_factor_sector", DataType::Float32, false),
        Field::new("pd", DataType::Float32, false),
        Field::new("lgd_mean", DataType::Float32, false),
        Field::new("exposure", DataType::Float32, false),
    ]))
}

pub fn write_simulation_results_to_parquet(
    trial_results: &[TrialResult],
    output_path: &str,
    sector_names: &[String],
    batch_size: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = ParquetStorage::new(output_path, batch_size.unwrap_or(10000))?;
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

        storage.batch_buffer.push(AssetRecord {
            trial_id: 0,
            asset_id: 1,
            sector_id: 0,
            sector_name: "Tech".to_string(),
            defaulted: true,
            time_to_default: Some(0.5),
            loss_amount: 1000.0,
            recovery_rate: 0.4,
            asset_value: -2.0,
            systematic_factor_global: -1.5,
            systematic_factor_sector: -0.8,
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
}
