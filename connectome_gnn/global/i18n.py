"""Internationalization support for global deployment."""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

from ..robust.logging_config import get_logger


@dataclass
class LocaleInfo:
    """Information about a locale."""
    code: str
    name: str
    native_name: str
    language: str
    country: str
    rtl: bool = False  # Right-to-left text
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"  # Number formatting locale
    currency_code: str = "USD"


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self, default_locale: str = "en_US", 
                 translations_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("i18n_manager")
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Setup translations directory
        if translations_dir:
            self.translations_dir = Path(translations_dir)
        else:
            self.translations_dir = Path(__file__).parent / "translations"
        
        # Initialize storage
        self.translations = {}
        self.locales = {}
        
        # Load built-in locales
        self._load_builtin_locales()
        
        # Load translations
        self._load_translations()
    
    def _load_builtin_locales(self):
        """Load built-in locale information."""
        builtin_locales = {
            "en_US": LocaleInfo(
                code="en_US",
                name="English (United States)",
                native_name="English (United States)",
                language="en",
                country="US",
                rtl=False,
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="en_US",
                currency_code="USD"
            ),
            "es_ES": LocaleInfo(
                code="es_ES",
                name="Spanish (Spain)",
                native_name="Español (España)",
                language="es",
                country="ES",
                rtl=False,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="es_ES",
                currency_code="EUR"
            ),
            "fr_FR": LocaleInfo(
                code="fr_FR",
                name="French (France)",
                native_name="Français (France)",
                language="fr",
                country="FR",
                rtl=False,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="fr_FR",
                currency_code="EUR"
            ),
            "de_DE": LocaleInfo(
                code="de_DE",
                name="German (Germany)",
                native_name="Deutsch (Deutschland)",
                language="de",
                country="DE",
                rtl=False,
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="de_DE",
                currency_code="EUR"
            ),
            "ja_JP": LocaleInfo(
                code="ja_JP",
                name="Japanese (Japan)",
                native_name="日本語 (日本)",
                language="ja",
                country="JP",
                rtl=False,
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                number_format="ja_JP",
                currency_code="JPY"
            ),
            "zh_CN": LocaleInfo(
                code="zh_CN",
                name="Chinese (China)",
                native_name="中文 (中国)",
                language="zh",
                country="CN",
                rtl=False,
                date_format="%Y-%m-%d",
                time_format="%H:%M",
                number_format="zh_CN",
                currency_code="CNY"
            )
        }
        
        self.locales.update(builtin_locales)
    
    def _load_translations(self):
        """Load translation files."""
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
        
        # Load translation files
        for locale_code in self.locales.keys():
            translation_file = self.translations_dir / f"{locale_code}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[locale_code] = json.load(f)
                    self.logger.debug(f"Loaded translations for {locale_code}")
                except Exception as e:
                    self.logger.error(f"Failed to load translations for {locale_code}: {e}")
                    self.translations[locale_code] = {}
            else:
                self.translations[locale_code] = {}
    
    def _create_default_translations(self):
        """Create default translation files."""
        default_translations = {
            "en_US": {
                "common": {
                    "yes": "Yes",
                    "no": "No",
                    "ok": "OK",
                    "cancel": "Cancel",
                    "save": "Save",
                    "load": "Load",
                    "delete": "Delete",
                    "edit": "Edit",
                    "create": "Create",
                    "error": "Error",
                    "warning": "Warning",
                    "info": "Information",
                    "success": "Success"
                },
                "connectome": {
                    "model": "Model",
                    "training": "Training",
                    "dataset": "Dataset",
                    "validation": "Validation",
                    "performance": "Performance",
                    "metrics": "Metrics",
                    "epoch": "Epoch",
                    "batch": "Batch",
                    "loss": "Loss",
                    "accuracy": "Accuracy"
                },
                "messages": {
                    "model_loaded": "Model loaded successfully",
                    "training_started": "Training started",
                    "training_completed": "Training completed",
                    "validation_failed": "Validation failed",
                    "file_not_found": "File not found",
                    "permission_denied": "Permission denied",
                    "invalid_input": "Invalid input"
                }
            },
            "es_ES": {
                "common": {
                    "yes": "Sí",
                    "no": "No",
                    "ok": "Aceptar",
                    "cancel": "Cancelar",
                    "save": "Guardar",
                    "load": "Cargar",
                    "delete": "Eliminar",
                    "edit": "Editar",
                    "create": "Crear",
                    "error": "Error",
                    "warning": "Advertencia",
                    "info": "Información",
                    "success": "Éxito"
                },
                "connectome": {
                    "model": "Modelo",
                    "training": "Entrenamiento",
                    "dataset": "Conjunto de datos",
                    "validation": "Validación",
                    "performance": "Rendimiento",
                    "metrics": "Métricas",
                    "epoch": "Época",
                    "batch": "Lote",
                    "loss": "Pérdida",
                    "accuracy": "Precisión"
                },
                "messages": {
                    "model_loaded": "Modelo cargado exitosamente",
                    "training_started": "Entrenamiento iniciado",
                    "training_completed": "Entrenamiento completado",
                    "validation_failed": "Validación falló",
                    "file_not_found": "Archivo no encontrado",
                    "permission_denied": "Permiso denegado",
                    "invalid_input": "Entrada inválida"
                }
            },
            "fr_FR": {
                "common": {
                    "yes": "Oui",
                    "no": "Non",
                    "ok": "OK",
                    "cancel": "Annuler",
                    "save": "Enregistrer",
                    "load": "Charger",
                    "delete": "Supprimer",
                    "edit": "Modifier",
                    "create": "Créer",
                    "error": "Erreur",
                    "warning": "Avertissement",
                    "info": "Information",
                    "success": "Succès"
                },
                "connectome": {
                    "model": "Modèle",
                    "training": "Entraînement",
                    "dataset": "Jeu de données",
                    "validation": "Validation",
                    "performance": "Performance",
                    "metrics": "Métriques",
                    "epoch": "Époque",
                    "batch": "Lot",
                    "loss": "Perte",
                    "accuracy": "Précision"
                },
                "messages": {
                    "model_loaded": "Modèle chargé avec succès",
                    "training_started": "Entraînement commencé",
                    "training_completed": "Entraînement terminé",
                    "validation_failed": "Validation échouée",
                    "file_not_found": "Fichier non trouvé",
                    "permission_denied": "Permission refusée",
                    "invalid_input": "Entrée invalide"
                }
            }
        }
        
        # Save translation files
        for locale_code, translations in default_translations.items():
            translation_file = self.translations_dir / f"{locale_code}.json"
            try:
                with open(translation_file, 'w', encoding='utf-8') as f:
                    json.dump(translations, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Created default translations for {locale_code}")
            except Exception as e:
                self.logger.error(f"Failed to create translations for {locale_code}: {e}")
    
    def set_locale(self, locale_code: str) -> bool:
        """Set the current locale."""
        if locale_code in self.locales:
            self.current_locale = locale_code
            self.logger.info(f"Locale set to {locale_code}")
            return True
        else:
            self.logger.warning(f"Locale {locale_code} not available")
            return False
    
    def get_locale(self) -> str:
        """Get the current locale."""
        return self.current_locale
    
    def get_locale_info(self, locale_code: str = None) -> Optional[LocaleInfo]:
        """Get information about a locale."""
        code = locale_code or self.current_locale
        return self.locales.get(code)
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locales."""
        return list(self.locales.keys())
    
    def translate(self, key: str, locale: str = None, default: str = None, **kwargs) -> str:
        """Translate a key to the current or specified locale."""
        target_locale = locale or self.current_locale
        
        # Get translations for target locale
        locale_translations = self.translations.get(target_locale, {})
        
        # Navigate through nested keys (e.g., "common.yes")
        keys = key.split('.')
        current = locale_translations
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                # Key not found, try default locale
                if target_locale != self.default_locale:
                    return self.translate(key, self.default_locale, default, **kwargs)
                else:
                    # Return default or key itself
                    return default or key
        
        # Apply string formatting if kwargs provided
        if kwargs and isinstance(current, str):
            try:
                return current.format(**kwargs)
            except (KeyError, ValueError):
                return current
        
        return str(current)
    
    def t(self, key: str, **kwargs) -> str:
        """Short alias for translate."""
        return self.translate(key, **kwargs)
    
    def add_translation(self, locale: str, key: str, value: str, save: bool = True):
        """Add or update a translation."""
        if locale not in self.translations:
            self.translations[locale] = {}
        
        # Navigate to nested location
        keys = key.split('.')
        current = self.translations[locale]
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        
        # Save to file if requested
        if save:
            self._save_translations(locale)
    
    def _save_translations(self, locale: str):
        """Save translations for a locale to file."""
        translation_file = self.translations_dir / f"{locale}.json"
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[locale], f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved translations for {locale}")
        except Exception as e:
            self.logger.error(f"Failed to save translations for {locale}: {e}")
    
    def format_number(self, number: float, locale: str = None) -> str:
        """Format a number according to locale conventions."""
        target_locale = locale or self.current_locale
        locale_info = self.get_locale_info(target_locale)
        
        if not locale_info:
            return str(number)
        
        # Simple number formatting (would use locale module in production)
        if locale_info.country in ["US", "GB"]:
            return f"{number:,.2f}"  # 1,234.56
        elif locale_info.country in ["DE", "FR", "ES"]:
            # European format: 1.234,56
            formatted = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return formatted
        else:
            return f"{number:.2f}"
    
    def format_currency(self, amount: float, locale: str = None) -> str:
        """Format currency according to locale conventions."""
        target_locale = locale or self.current_locale
        locale_info = self.get_locale_info(target_locale)
        
        if not locale_info:
            return f"${amount:.2f}"
        
        # Format number
        formatted_number = self.format_number(amount, target_locale)
        
        # Add currency symbol/code
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥"
        }
        
        symbol = currency_symbols.get(locale_info.currency_code, locale_info.currency_code)
        
        # Position symbol based on locale
        if locale_info.country == "US":
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number} {symbol}"
    
    def get_text_direction(self, locale: str = None) -> str:
        """Get text direction for locale (ltr or rtl)."""
        target_locale = locale or self.current_locale
        locale_info = self.get_locale_info(target_locale)
        
        if locale_info and locale_info.rtl:
            return "rtl"
        else:
            return "ltr"


# Global i18n manager instance
_global_i18n_manager = None

def get_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager


def _(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_i18n_manager().translate(key, **kwargs)