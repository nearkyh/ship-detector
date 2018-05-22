## Convert xml to csv file
- `python xml_to_csv.py`

## Convert to TFRecord format
- `python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record`
- `python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record`
