# Rapport pré-modification — Point 1 des propositions

## Périmètre analysé
- `Serialization/src/serializer.cpp`
- `Serialization/data/serializer.hpp`
- `Main/src/main.cpp`
- `Env/src/environment.cpp`
- `Entities/src/aircraft.cpp`
- `Entities/src/sam.cpp`
- `Initializer/src/checkpointinitializersystem.cpp`
- `PPO/data/trainingagent_def.hpp`
- `resources/settings/*.json`

## Constats précis
1. Le parse JSON est direct (`json::parse(std::ifstream(file))`) sans gestion explicite des erreurs d'ouverture/parse.
2. Le chargement des configs enregistrées (`Serializer::Load`) ne distingue pas warnings et erreurs bloquantes.
3. Il n'existe pas de contrat de version (`schemaVersion`) vérifié de façon centralisée.
4. Plusieurs consommateurs lisent les clés de manière stricte, ce qui peut lever des exceptions tardives.

## Proposition d'implémentation (itération 1)
1. Introduire une couche centralisée dans `Serializer`:
   - Validation de contrat `schemaVersion` (version attendue = 1).
   - Fallback si la clé est absente (warning + valeur par défaut 1).
   - Rejet si type invalide ou version incompatible (erreur bloquante pour le fichier).
2. Rendre les erreurs actionnables via logs structurés:
   - codes: `CFG_MISSING_FILE`, `CFG_PARSE_FAILURE`, `CFG_SCHEMA_MISSING`, `CFG_SCHEMA_INVALID_TYPE`, `CFG_SCHEMA_MISMATCH`, `CFG_LOAD_EXCEPTION`.
   - contexte: fichier + détail.
3. Sécuriser `Parse`:
   - vérifier l'ouverture de fichier.
   - capturer les exceptions de parsing.
4. Garantir la persistance du contrat en sortie:
   - injecter `schemaVersion=1` à la sauvegarde si absent.

## Impact attendu
- Démarrage plus robuste sur fichier absent/corrompu.
- Diagnostic de config plus rapide.
- Transition progressive vers un schéma versionné sans casser les fichiers existants.
