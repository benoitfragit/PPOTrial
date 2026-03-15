# Proposition d'amélioration technique (priorisée)

Ce document synthétise les axes d'amélioration techniques identifiés dans le dépôt, classés par ordre de priorité.

## 1) Fiabiliser la configuration et la sérialisation (priorité critique)

### Constats
- Le parse JSON est direct (`json::parse(std::ifstream(file))`) sans couche de validation unifiée ni gestion d'erreur explicite.
- Plusieurs lectures de configuration accèdent aux clés de manière stricte (sans fallback/versionning), ce qui fragilise le démarrage si un fichier est partiellement invalide.
- Certains cas invalides sont ignorés silencieusement (entité incomplète, scénario absent), ce qui rend le diagnostic difficile.

### Propositions
1. **Introduire un contrat de config versionné**
   - Ajouter un champ `schemaVersion` dans chaque fichier de settings.
   - Valider la version avant chargement métier.
2. **Créer une couche de chargement robuste**
   - `ConfigLoader` centralisé: lecture, validation, defaults, rapport d'anomalies.
   - Séparer les erreurs bloquantes (arrêt) des warnings (valeurs par défaut appliquées).
3. **Rendre les erreurs actionnables**
   - Logs structurés: fichier, clé, valeur attendue, fallback utilisé.
   - Résumé de boot: fichiers chargés, fichiers ignorés, defaults appliqués.

### Gains attendus
- Démarrage plus fiable, debug accéléré, baisse des régressions liées à la config.

---

## 2) Stabiliser la boucle de simulation et le timing runtime

### Constats
- Cadence runtime pilotée par un `sleep_for(2ms)` fixe, potentiellement désaligné du `TimeStep` logique.
- Arrêt du thread potentiellement non déterministe selon l'ordre de traitement des commandes et du flag d'exécution.

### Propositions
1. **Séparer tick simulation et tick rendu**
   - Simulation en `dt` fixe (accumulateur temps).
   - Rendu découplé et adaptatif.
2. **Arrêt déterministe**
   - Protocole `request stop -> ack STOPPED -> join`.
   - Éviter les transitions d'état ambiguës.
3. **Instrumentation runtime**
   - Mesurer durée moyenne/p95 du tick simulation.
   - Alerte en cas de dérive de cadence.

### Gains attendus
- Reproductibilité des runs RL, moins de bugs timing, meilleure comparabilité expérimentale.

---

## 3) Optimiser les sections O(N²) pour la montée en charge

### Constats
- Certaines interactions sont évaluées en double boucle (entité vs toutes les autres).
- Clustering SAM basé sur comparaisons de paires complètes.

### Propositions
1. **Ajouter une indexation spatiale (grid / hash 2D)**
   - Pré-filtrage des candidats proches.
2. **Pipeline broad-phase / narrow-phase**
   - Broad-phase spatial, narrow-phase physique/probabiliste.
3. **Optimiser le clustering des menaces**
   - Réduire les paires candidates avant union-find.

### Gains attendus
- Meilleure tenue en charge, FPS plus stables, entraînement plus rapide à scénarios denses.

---

## 4) Réduire le coût du mécanisme de snapshot

### Constats
- Snapshot reconstruit à chaque boucle avec copie manuelle de plusieurs composantes.
- Risque de coût CPU/mémoire élevé et de dette d'évolution (oubli de nouveaux composants).

### Propositions
1. **Snapshot incrémental**
   - Copier seulement les entités/composants modifiés.
2. **View model dédié au rendu/UI**
   - Publier une structure compacte orientée affichage.
3. **Contrôle de couverture**
   - Vérification automatique des composants exposés au renderer.

### Gains attendus
- Réduction de la charge runtime, architecture plus robuste aux évolutions.

---

## 5) MLOps: clarifier le cycle de vie des modèles PPO

### Constats
- Le flux `eval/prod` peut laisser `prod` obsolète selon la logique actuelle de création de dossier.
- Gestion de seed trop statique pour certains usages d'expérimentation.

### Propositions
1. **Politique de promotion explicite**
   - Toujours sauver `latest-eval`.
   - Promouvoir vers `prod` uniquement après critère d'évaluation.
2. **Traçabilité des runs**
   - Sauver metadata: seed, version config, commit, date.
3. **Modes seed configurables**
   - Mode déterministe (debug/repro), mode variable contrôlé (exploration).

### Gains attendus
- Déploiement de modèle plus sûr, rollback simple, meilleure reproductibilité scientifique.

---

## 6) Industrialiser le build (portabilité et CI)

### Constats
- Configuration de build fortement orientée environnement Windows/VS.
- Dépendances non totalement abstraites par plateforme.

### Propositions
1. **Paramétrer les dépendances par OS/compilateur**
   - Variables/presets par cible.
2. **Adopter des presets CMake**
   - Profils dev/ci clairs (debug/release, OS, toolchain).
3. **Mettre une CI minimale multi-plateforme**
   - Build matrix + smoke test headless.

### Gains attendus
- Onboarding simplifié, meilleure reproductibilité, baisse du risque d'intégration.

---

## 7) Renforcer l'observabilité technique et métier

### Constats
- Plusieurs chemins d'échec/skip sont silencieux.
- Peu d'indicateurs consolidés de santé au boot et en runtime.

### Propositions
1. **Logs structurés avec codes d'événements**
   - Ex: `CFG_MISSING_FILE`, `SCENARIO_ENTITY_SKIPPED`, etc.
2. **Tableau de santé au démarrage**
   - Nombre d'entités chargées, agents créés, mappings résolus.
3. **Métriques runtime**
   - Tick rate, temps boucle, entités actives/détruites, erreurs cumulées.

### Gains attendus
- MTTR réduit, supervision facilitée, exploitation plus fiable.

---

## Plan d'exécution recommandé (12 semaines)

1. **S1-S2**: hardening config + logs de boot.
2. **S3-S4**: refonte timing simulation (dt fixe + arrêt déterministe).
3. **S5-S6**: optimisation spatiale des interactions critiques.
4. **S7-S8**: snapshot incrémental / contrat rendu.
5. **S9-S10**: politique MLOps de promotion modèle.
6. **S11-S12**: presets CMake + CI multi-plateforme.

## Critères de succès transverses
- Aucune régression fonctionnelle sur scénarios existants.
- Diminution mesurable du temps moyen de tick.
- Diminution des incidents de démarrage liés à la config.
- Pipeline build reproductible sur au moins 2 environnements.
