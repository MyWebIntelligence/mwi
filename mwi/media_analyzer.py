import io
import hashlib
import aiohttp
import json
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from typing import Dict, Any, Optional, List

def generer_palette_web_safe():
    """
    Génère les 216 couleurs RGB de la palette Web Safe.
    """
    niveaux = [0, 51, 102, 153, 204, 255]
    return [(r, g, b) for r in niveaux for g in niveaux for b in niveaux]

def distance_rgb(c1, c2):
    """
    Calcule la distance euclidienne au carré entre deux couleurs RGB.
    """
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

def convertir_vers_web_safe(rgb):
    """
    Convertit une couleur RGB en sa correspondante la plus proche dans la palette Web Safe.
    """
    palette = generer_palette_web_safe()
    return min(palette, key=lambda c: distance_rgb(rgb, c))

class MediaAnalyzer:
    """Analyseur de médias avec traitement asynchrone"""
    
    def __init__(self, session: aiohttp.ClientSession, settings: Dict[str, Any]):
        self.session = session
        self.settings = settings
        self.max_size = self.settings.get('media_max_file_size', 10 * 1024 * 1024)

    async def analyze_image(self, url: str) -> Dict[str, Any]:
        """Analyse complète d'une image"""
        result = {
            'error': None,
            'width': None,
            'height': None,
            'format': None,
            'file_size': None,
            'color_mode': None,
            'has_transparency': False,
            'aspect_ratio': None,
            'exif_data': None,
            'image_hash': None,
            'dominant_colors': [],
            'websafe_colors': []
        }

        try:
            # Téléchargement avec limite de taille
            async with self.session.get(url) as response:
                if response.content_length and response.content_length > self.max_size:
                    raise ValueError(f"Taille dépassée ({response.content_length} bytes)")
                
                content = await response.read()
                if len(content) > self.max_size:
                    raise ValueError(f"Taille réelle dépassée ({len(content)} bytes)")

                result['file_size'] = len(content)
                
                # Hash perceptuel
                result['image_hash'] = hashlib.sha256(content).hexdigest()
                
                # Analyse avec PIL
                with Image.open(io.BytesIO(content)) as img:
                    self._analyze_image_properties(img, result)
                    self._extract_colors(img, result)
                    self._extract_exif(img, result)

        except Exception as e:
            result['error'] = str(e)
        
        return result

    def _analyze_image_properties(self, img: Image.Image, result: Dict):
        """Extrait les propriétés basiques de l'image"""
        result.update({
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'color_mode': img.mode,
            'has_transparency': self._has_transparency(img),
            'aspect_ratio': round(img.width / img.height, 2) if img.height else 0
        })

    def _has_transparency(self, img: Image.Image) -> bool:
        """Détecte la transparence"""
        if img.mode in ('RGBA', 'LA'):
            alpha = img.getchannel('A')
            return any(pixel < 255 for pixel in alpha.getdata())
        return False

    def _extract_colors(self, img: Image.Image, result: Dict[str, Any]):
        """Extrait les couleurs dominantes avec K-means"""
        n_colors = self.settings.get('media_n_dominant_colors', 5)
        try:
            # Réduire la taille pour le traitement
            img = img.resize((100, 100)).convert('RGB')
            pixels = np.array(img).reshape(-1, 3)
            
            # Clustering
            kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42)
            kmeans.fit(pixels)
            
            # Compter les occurrences
            counts = np.bincount(kmeans.labels_)
            total = sum(counts)
            
            # Tri par fréquence
            sorted_colors = sorted(zip(kmeans.cluster_centers_, counts),
                                 key=lambda x: x[1], reverse=True)
            
            # Formatage des résultats
            dominant_colors = [{
                'rgb': tuple(map(int, color)),
                'percentage': round(count / total * 100, 2)
            } for color, count in sorted_colors]
            
            result['dominant_colors'] = dominant_colors
            
            # Conversion en couleurs web safe
            websafe_palette = {}
            for item in dominant_colors:
                websafe_color = convertir_vers_web_safe(item['rgb'])
                websafe_hex = '#%02x%02x%02x' % websafe_color
                if websafe_hex not in websafe_palette:
                    websafe_palette[websafe_hex] = 0
                websafe_palette[websafe_hex] += item['percentage']
            
            result['websafe_colors'] = websafe_palette

        except Exception as e:
            result['error'] = f"Erreur d'analyse couleur: {str(e)}"

    def _extract_exif(self, img: Image.Image, result: Dict[str, Any]):
        """Extrait les métadonnées EXIF"""
        try:
            exif_data = img.getexif()
            if exif_data:
                exif = {
                    "ImageWidth": exif_data.get(256),
                    "ImageLength": exif_data.get(257),
                    "Make": exif_data.get(271),
                    "Model": exif_data.get(272),
                    "DateTime": exif_data.get(306),
                }
                result['exif_data'] = {k: v for k, v in exif.items() if v is not None}
        except Exception as e:
            # Ne pas écraser une erreur précédente
            if not result.get('error'):
                result['error'] = f"Erreur EXIF: {str(e)}"
