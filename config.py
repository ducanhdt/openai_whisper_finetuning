class Config:
    learning_rate = 0.0001
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 5
    gradient_accumulation_steps = 1
    sample_rate = 16000
    log_output_dir = "content/logs"
    check_output_dir = "content/artifacts"
    train_name = "whisper"
    train_id = "00001"
    model_name = "tiny"
    lang = "en"
