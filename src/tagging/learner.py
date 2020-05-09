import torch
import numpy as np


class Learner:

    def __init__(self,
                 model,
                 learning_rate=0.0005,
                 weight_decay=0.01,
                 loss_function=torch.nn.MultiLabelSoftMarginLoss,
                 optimizer_function=torch.optim.Adam,
                 probability_threshold=0.5
                 ):

        self.model = model
        self.loss_function = loss_function()
        self.optimizer = optimizer_function(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.probability_threshold = probability_threshold
        self.epoch = None

    def learn(self, dl_train, dl_valid=None, epochs=5):

        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            metrics = self._run_epoch(dl_train, True)

            print(f"Epoch {epoch + 1}; training accuracy {metrics['accuracy']:0.2f} %")
            print(f"Epoch {epoch + 1}; training precision {metrics['precision']:0.2f} %")
            print(f"Epoch {epoch + 1}; average training loss {metrics['avg_loss']:0.3f}")

            if dl_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    metrics = self._run_epoch(dl_valid, False)

                print(f"Epoch {epoch + 1}; valid accuracy {metrics['accuracy']:0.2f} %")
                print(f"Epoch {epoch + 1}; valid precision {metrics['precision']:0.2f} %")
                print(f"Epoch {epoch + 1}; average valid loss {metrics['avg_loss']:0.3f}")

    def get_model(self):
        return self.model

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def _run_epoch(self, dl, update_weights=False):
        n_correct, n_all, n_true_positive, n_positive_all = 0, 0, 0, 0
        losses = []
        for i_batch, sample in enumerate(dl):
            images, tags = sample
            output = self.model(images)
            loss = self.loss_function(output, tags)
            losses.append(loss.item())

            y_true = tags.cpu().detach().numpy().astype(int)
            probabilities = torch.exp(output.cpu()).detach().numpy()
            y_pred = (probabilities > self.probability_threshold).astype(int)

            n_correct += np.sum(y_true == y_pred)
            n_all += y_true.size
            n_true_positive += np.sum((y_true == y_pred)[y_true == 1])
            n_positive_all += np.sum((y_true == 1))

            if update_weights:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {self.epoch + 1}; batch {i_batch}; loss {loss.item()}")

        metrics = {
            "accuracy": 100 * n_correct / n_all,
            "precision": 100 * n_true_positive / n_positive_all,
            "avg_loss": np.mean(losses),
        }

        return metrics
