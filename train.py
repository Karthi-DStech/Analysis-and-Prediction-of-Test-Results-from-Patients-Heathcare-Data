from call_methods import make_network, make_params
from utils.logs import Logger
from utils.preprocessing import DataProcessor
from utils.train_test_split import TrainTestProcessor
from options.train_options import TrainOptions
from utils.save_utils import save_model_and_logs
import os


def run() -> None:
    """
    Run the training process

    Parameters
    ----------
    None

    Returns
    -------
    None

    Process
    -------
    1. Parse the training options
    2. Initialize the logger
    3. Preprocess the data
    4. Split the data into training and testing sets
    5. Train the model
    6. Evaluate the model
    7. Tune the model
    8. Save the model and logs.
    """
    # Parse the training options
    opt = TrainOptions().parse()

    # Initialize the logger
    logger = Logger(opt)

    # Initialize and process data
    processor = DataProcessor(opt.data_path, logger, opt)

    # Missing Value Imputation Dictionary
    imputation_dict = {opt.missing_values_imputation}

    # Encode the data
    processed_data, missing_values = processor.process_and_save(
        imputation_dict,
        label_encode_columns=opt.label_encode_columns,
        one_hot_encode_columns=opt.one_hot_encode_columns,
        dtype_dict=opt.dtype_dict,
        feature_engg_names=opt.feature_engg_name,
    )

    # Log missing values
    logger.update_log("data_processing", "missing_values", missing_values.to_dict())

    # Initialize TrainTestProcessor
    train_test_processor = TrainTestProcessor(processed_data, logger, opt)

    X_train, X_test, y_train, y_test = train_test_processor.process()

    # Perform final checks
    train_test_processor.final_checks(X_train, X_test, y_train, y_test)

    # Initialize model using make_network
    model = make_network(opt.model_name, logger, opt)

    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Tune the model
    get_params_func = make_params(opt.model_name)
    model.model_tuning(
        get_params_func,
        X_train,
        y_train,
        X_test,
        y_test,
        n_trials=opt.n_trials,
    )

    # Save the model and logs
    save_model_and_logs(model, logger, opt)


if __name__ == "__main__":
    run()
