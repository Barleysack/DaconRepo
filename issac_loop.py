def confusion_matrix(true, pred, num):
    cm=np.array([[0 for _ in range(num)] for _ in range(num)])
    for i in range(len(true)):
        cm[true[i]][pred[i]]+=1
    return cm

def draw_confusion_matrix(true, pred, num,save_dir):
    cm = confusion_matrix(true, pred, num)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None],
                index=list(range(num)), columns=list(range(num)))    
    df = df.fillna(0)  # NaN 값을 0으로 변경
    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    if save_dir:
        save_folder=save_dir
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_folder+f"/confusion_matrix_{len(os.listdir(save_folder))}.png")
    plt.close('all')

def get_compute_metrics(num,save_dir=None):
    def compute_metrics(pred):
        """ validation을 위한 metrics function """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        f1 = f1_score(labels, preds, average="micro", labels=list(range(num))) * 100.0
        acc = accuracy_score(labels, preds)

        draw_confusion_matrix(labels,preds,num,save_dir)
        return {
            'micro f1 score': f1,
            'accuracy': acc,
        }
    return compute_metrics

def train():
    KST=timezone('Asia/Seoul')
    DATE=str(datetime.now().astimezone(KST))[:19]
    MODEL="klue/roberta-large"
    BATCH_SIZE=8
    LEARNING_RATE=1e-5
    EPOCHS=20
    OUTPUT='/'.join([root,'runs',DATE])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(2022)

    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    train,eval=get_dataset(root+'/data/train_data.csv',tokenizer,'train')
    test=get_dataset(root+'/data/test_data.csv',tokenizer,'test')

    config=AutoConfig.from_pretrained(MODEL)
    config.num_labels=3
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,config=config)
    model.to(device)

    training_args = TrainingArguments(
    output_dir=OUTPUT,                    # output directory
    #save_total_limit=2,                        # number of total save model.
    #save_steps=SAVE_STEPS,                     # model saving step.
    num_train_epochs=EPOCHS,                   # total number of training epochs
    learning_rate=LEARNING_RATE,               # learning_rate
    per_device_train_batch_size=BATCH_SIZE,    # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,     # batch size for evaluation
    weight_decay=0.01,                         # strength of weight decay
    logging_dir=OUTPUT+'/logs',                      # directory for storing logs
    logging_steps=100,                         # log saving step.
    save_strategy='epoch',
    evaluation_strategy='epoch',               # evaluation strategy to adopt during training
                                            # `no`: No evaluation during training.
                                            # `steps`: Evaluate every `eval_steps`.
                                            # `epoch`: Evaluate every end of epoch.
    load_best_model_at_end = True,
    metric_for_best_model = "micro f1 score",
    greater_is_better = True,
    report_to="wandb"
    )

    wandb_configs={'runs':DATE,'model':MODEL,'bsz':BATCH_SIZE,'lr':LEARNING_RATE,'epochs':EPOCHS,}
    run=wandb.init(project='Sentence classification',name=DATE,config=wandb_configs)
    os.makedirs(OUTPUT)
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train,
                    eval_dataset=eval,
                    compute_metrics=get_compute_metrics(config.num_labels,OUTPUT+'/confusion'),
                    )
    trainer.train()
    save_directory = OUTPUT+'/best'
    model.save_pretrained(save_directory)
    run.finish()

    prediction=trainer.predict(test)
    outs=pd.read_csv(root+'/data/test_data.csv')
    outs['label']=torch.argmax(prediction.predictions,axis=1).tolist()
    outs.to_csv(OUTPUT+'/result.csv')
