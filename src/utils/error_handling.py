"""Error handling utilities for EdgeFormer."""
import sys
import logging
import traceback

logger = logging.getLogger('edgeformer')

class EdgeFormerError(Exception):
    """Base exception class for EdgeFormer errors."""
    def __init__(self, message, code=None):
        self.message = message
        self.code = code
        super().__init__(self.message)
        
    def __str__(self):
        if self.code:
            return f"[Error {self.code}] {self.message}"
        return f"[Error] {self.message}"

class ConfigurationError(EdgeFormerError):
    """Raised when configuration is invalid."""
    pass

class MemoryError(EdgeFormerError):
    """Raised for associative memory issues."""
    pass

class ModelError(EdgeFormerError):
    """Raised for model-related issues."""
    pass

class OptimizationError(EdgeFormerError):
    """Raised for device optimization issues."""
    pass

def setup_exception_handler():
    """Set up global exception handler for more user-friendly errors."""
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, EdgeFormerError):
            logger.error(f"EdgeFormer error: {exc_value}")
            print(f"\nEdgeFormer error: {exc_value}")
            print("For more information, see the log file or run with --verbose")
        else:
            logger.error("Unexpected error", exc_info=(exc_type, exc_value, exc_traceback))
            print("\nAn unexpected error occurred:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("\nPlease report this issue at: https://github.com/oscarnunez/EdgeFormer/issues")
    
    sys.excepthook = global_exception_handler
    
def format_friendly_error(error_message, suggestion=None):
    """Format an error message with suggestion."""
    friendly_message = f"Error: {error_message}"
    if suggestion:
        friendly_message += f"\n\nSuggestion: {suggestion}"
    return friendly_message