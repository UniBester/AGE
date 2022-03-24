from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'af_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['af_train'],
		'train_target_root': dataset_paths['af_train'],
		'valid_source_root': dataset_paths['af_valid'],
		'valid_target_root': dataset_paths['af_valid'],
	},
	'fl_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['fl_train'],
		'train_target_root': dataset_paths['fl_train'],
		'valid_source_root': dataset_paths['fl_valid'],
		'valid_target_root': dataset_paths['fl_valid'],
	},
}
