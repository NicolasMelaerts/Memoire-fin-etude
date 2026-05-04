"""
shared - Code partagé entre toutes les expériences.

Ce package contient :
  - model.py      : architecture du réseau SimpleCNN et fonctions GradCAM
  - config.py     : hyperparamètres et configuration
  - dataset.py    : chargement et préparation des données
  - trainer.py    : Trainer unifié pour toutes les stratégies
  - strategies.py : Stratégies d'entraînement (Normal, Double BP, GradCAM, GAIN)

Usage :
    from shared.trainer import Trainer
    from shared.strategies import NormalStrategy, GradCAMStrategy, ...

    trainer = Trainer(NormalStrategy())
    history, model, duration = trainer.run(train_loader, test_loader)
"""

__all__ = [
    'model',
    'config',
    'dataset',
    'trainer',
    'strategies',
]
