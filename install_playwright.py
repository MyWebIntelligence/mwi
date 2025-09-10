#!/usr/bin/env python3
"""
Script d'installation pour Playwright
Installe les navigateurs nécessaires pour l'extraction dynamique des médias
"""
import subprocess
import sys

def install_playwright_browsers():
    """
    Installe les navigateurs Playwright nécessaires
    """
    try:
        print("Installation des navigateurs Playwright...")
        result = subprocess.run([
            sys.executable, "-m", "playwright", "install", "chromium"
        ], check=True, capture_output=True, text=True)
        print("✅ Installation des navigateurs Playwright réussie")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'installation des navigateurs Playwright:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Playwright n'est pas installé. Veuillez d'abord installer les dépendances avec:")
        print("pip install -r requirements.txt")
        return False

def check_playwright_available():
    """
    Vérifie si Playwright est disponible
    """
    try:
        import playwright
        print("✅ Playwright est disponible")
        return True
    except ImportError:
        print("❌ Playwright n'est pas installé")
        return False

if __name__ == "__main__":
    print("=== Installation de Playwright pour MyWebIntelligence ===")
    
    if check_playwright_available():
        if install_playwright_browsers():
            print("\n🎉 Installation terminée avec succès!")
            print("L'extraction dynamique des médias est maintenant disponible.")
        else:
            print("\n⚠️  Installation des navigateurs échouée.")
            sys.exit(1)
    else:
        print("\nVeuillez d'abord installer Playwright avec:")
        print("pip install -r requirements.txt")
        sys.exit(1)
