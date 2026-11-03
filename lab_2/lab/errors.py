"""Ошибки обработки данных."""


class Error(Exception):
    """Базовая ошибка."""


class LoadError(Error):
    """Ошибка загрузки файла."""


class SchemaError(Error):
    """Ошибка структуры данных."""
