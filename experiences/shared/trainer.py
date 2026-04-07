"""
trainer.py — Trainer unifié pour toutes les stratégies d'entraînement.

Simplifie l'entraînement en centralisant la boucle train/validation.
Chaque stratégie définit uniquement comment calculer sa loss.

Usage:
    from shared.trainer import Trainer
    from shared.strategies import NormalStrategy

    trainer = Trainer(NormalStrategy())
    history, model, duration = trainer.run(train_loader, test_loader)
"""
import time
import torch
import torch.optim as optim

from .config import DEVICE, EPOCHS, LEARNING_RATE
from .model import get_fresh_model


class Trainer:
    """
    Trainer unifié compatible avec toutes les stratégies.

    Gère la boucle d'entraînement complète :
    - Initialisation du modèle et de l'optimizer
    - Boucle train/validation sur EPOCHS
    - Tracking de l'historique (loss, accuracy, métriques custom)
    - Mesure de la durée d'entraînement
    """

    def __init__(self, strategy, epochs=None, learning_rate=None, device=None, verbose=True):
        """
        Args:
            strategy: Instance d'une classe héritant de BaseStrategy
            epochs: Nombre d'epochs (défaut: EPOCHS de config)
            learning_rate: Learning rate (défaut: LEARNING_RATE de config)
            device: Device PyTorch (défaut: DEVICE de config)
            verbose: Afficher les logs pendant l'entraînement
        """
        self.strategy = strategy
        self.epochs = epochs or EPOCHS
        self.learning_rate = learning_rate or LEARNING_RATE
        self.device = device or DEVICE
        self.verbose = verbose

    def run(self, train_loader, test_loader):
        """
        Lance l'entraînement complet.

        Args:
            train_loader: DataLoader pour l'entraînement
            test_loader: DataLoader pour la validation

        Returns:
            history: dict avec les courbes d'apprentissage
                {
                    'train_loss': [...],
                    'train_acc': [...],
                    'test_loss': [...],  # Évalué sur le test set
                    'test_acc': [...],   # Évalué sur le test set
                    'train_ce': [...],
                    + métriques additionnelles selon la stratégie
                }
            model: Modèle entraîné (SimpleCNN)
            duration: Temps d'entraînement en secondes
        """
        # Initialisation
        model = get_fresh_model(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Historique des métriques
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],  # Évalué sur le test set (25% du dataset)
            'test_acc': [],   # Évalué sur le test set (25% du dataset)
            'train_ce': [],   # CE pure (sans pénalités) pour comparaison équitable
        }

        # Ajouter les métriques spécifiques à la stratégie
        for metric_name in self.strategy.get_tracking_metrics():
            history[metric_name] = []

        t0 = time.time()

        # Boucle d'entraînement
        for epoch in range(self.epochs):
            # --- Phase d'entraînement ---
            train_metrics = self._train_epoch(model, train_loader, optimizer)

            # --- Phase de validation ---
            val_metrics = self._validate_epoch(model, test_loader)

            # Enregistrer l'historique
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['train_ce'].append(train_metrics['ce_loss'])
            history['test_loss'].append(val_metrics['loss'])
            history['test_acc'].append(val_metrics['acc'])

            # Métriques additionnelles (grad_penalty, loc_loss, etc.)
            for metric_name in self.strategy.get_tracking_metrics():
                if metric_name in train_metrics:
                    history[metric_name].append(train_metrics[metric_name])

            # Affichage
            if self.verbose:
                self._print_epoch_summary(epoch, train_metrics, val_metrics)

        duration = time.time() - t0
        return history, model, duration

    def _train_epoch(self, model, train_loader, optimizer):
        """Exécute une epoch d'entraînement."""
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        correct = 0
        total = 0

        # Métriques additionnelles
        additional_metrics = {name: 0.0 for name in self.strategy.get_tracking_metrics()}

        for images, labels, heatmaps, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            heatmaps = heatmaps.to(self.device)

            optimizer.zero_grad()

            # La stratégie calcule la loss totale et retourne les métriques
            loss, metrics = self.strategy.compute_loss(model, images, labels, heatmaps)

            loss.backward()
            optimizer.step()

            # Accumulation des métriques
            total_loss += loss.item()
            total_ce += metrics['ce_loss']

            # Accuracy
            logits = metrics['logits']
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Métriques additionnelles (grad_penalty, loc_loss, etc.)
            for metric_name in self.strategy.get_tracking_metrics():
                if metric_name in metrics:
                    additional_metrics[metric_name] += metrics[metric_name]

        # Moyennes
        avg_metrics = {
            'loss': total_loss / len(train_loader),
            'ce_loss': total_ce / len(train_loader),
            'acc': 100.0 * correct / total,
        }

        for metric_name in self.strategy.get_tracking_metrics():
            avg_metrics[metric_name] = additional_metrics[metric_name] / len(train_loader)

        return avg_metrics

    def _validate_epoch(self, model, test_loader):
        """Évalue le modèle sur le test set."""
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _, _ in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            'loss': total_loss / len(test_loader),
            'acc': 100.0 * correct / total,
        }

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Affiche un résumé de l'epoch."""
        # Formatage de base
        msg = (f"  [{self.strategy.name}]  "
               f"Epoch {epoch+1:02d}/{self.epochs}  "
               f"train_loss={train_metrics['loss']:.4f}  "
               f"train_acc={train_metrics['acc']:.1f}%  "
               f"test_loss={val_metrics['loss']:.4f}  "
               f"test_acc={val_metrics['acc']:.1f}%")

        # Ajouter les métriques additionnelles si présentes
        for metric_name in self.strategy.get_tracking_metrics():
            if metric_name in train_metrics:
                msg += f"  {metric_name}={train_metrics[metric_name]:.4f}"

        print(msg)
