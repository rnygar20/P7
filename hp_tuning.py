import optuna
from GCN import run_GCN_hyp
from GraphSAGE import run_GraphSAGE_hyp
from download_dataset import get_dataset

def objective(trail):

    lr = trail.suggest_float("lr", 1e-5, 1e-1)
    hidden_channels = trail.suggest_int("hidden_channels", 16, 128)
    batch_size = trail.suggest_int("batch_size", 64, 512)
    size_gcn = trail.suggest_int("size_gcn", 1,5)
    weight_decay = trail.suggest_float("weight_decay",5e-4, 5e-2)
    epochs = trail.suggest_int("epochs", 50, 500)
    dropout = trail.suggest_float("dropout", 0.1, 0.9)

    val_loss = run_GraphSAGE_hyp(get_dataset('Cora'),lr,hidden_channels,size_gcn,weight_decay,epochs,dropout,batch_size)
    print(val_loss)
    return val_loss

def perform_hyperparameter_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))


    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial


if __name__ == "__main__":
    perform_hyperparameter_optimization()
