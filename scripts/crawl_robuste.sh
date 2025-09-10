#!/bin/bash

MAX_RETRIES=3
LOG_FILE="crawl_errors.log"

success_count=0
fail_count=0

try_block() {
    echo "Début du tour $1 - Tentative $2"
    python mywi.py land crawl --name=giletsjaunes --depth=0 --limit=100 2>&1
    return $?
}

catch_block() {
    echo "[ERREUR] Tour $1 échoué, nouvelle tentative dans $2 secondes..."
    sleep $2
}

for ((i=1; i<=230; i++)); do
    attempt=1
    success=0
    
    while [ $attempt -le $MAX_RETRIES ]; do
        if try_block $i $attempt; then
            success=1
            break
        else
            catch_block $i $((attempt * 2))
            ((attempt++))
        fi
    done

    if [ $success -eq 1 ]; then
        ((success_count++))
        echo "Tour $i réussi après $((attempt)) tentatives" >> "$LOG_FILE"
    else
        ((fail_count++))
        echo "ECHEC CRITIQUE: Tour $i après $MAX_RETRIES tentatives" >> "$LOG_FILE"
    fi

    sleep 1  # Pause de 1 seconde entre les tours
    echo "Fin du tour $i"
    echo "-----------------------------------"
    echo ""  # Ligne vide pour la lisibilité
    echo "Logs du tour $i enregistrés dans $LOG_FILE"
 
done

echo "Rapport final:"
echo "Tours réussis: $success_count"
echo "Tours échoués: $fail_count"
echo "Logs détaillés disponibles dans: $LOG_FILE"
