from utilities import save_cli_args,get_args,save_test_metrics
import options
from pytorch_tabular import TabularModel
import pandas as pd

parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = get_args()


TEST_DATASET_PATH = args.test_set_input_dir
MODEL_WEIGHTS = args.model_weights

test = pd.read_parquet(TEST_DATASET_PATH)


if __name__ == "__main__":
    
    loaded_model = TabularModel.load_model(MODEL_WEIGHTS)
    result = loaded_model.evaluate(test)
    print(result)
    print(f'Test Accuracy is result', result[0]['test_accuracy'])
    save_cli_args(args, TEST_OUTPUT_DIR, mode='tabular test')
    save_test_metrics(float(result[0]['test_loss']), TEST_OUTPUT_DIR)
