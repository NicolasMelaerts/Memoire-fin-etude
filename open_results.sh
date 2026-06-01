#!/bin/bash
# Ouvre tous les fichiers HTML de résultats dans le navigateur par défaut

ROOT="$(cd "$(dirname "$0")" && pwd)"

open "$ROOT/dataset_creator/index.html"
open "$ROOT/experiences/Résultats/exp0/exp0.html"
open "$ROOT/experiences/Résultats/exp1/exp1.html"
open "$ROOT/experiences/Résultats/exp2/exp2.html"
open "$ROOT/experiences/Résultats/exp3/exp3.html"
open "$ROOT/experiences/Résultats/exp4/exp4.html"
open "$ROOT/experiences/Résultats/multidataset/index.html"
open "$ROOT/real_datasets/oiseaux/results/cub_results.html"
open "$ROOT/real_datasets/poissons/results/fish_results.html"
open "$ROOT/real_datasets/poissons/results_efficacite/fish_results_2.html"