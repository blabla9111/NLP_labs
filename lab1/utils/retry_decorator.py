import time
from typing import Callable, Any, Tuple
from functools import wraps

def retry_on_failure(
    max_retries: int = 3, 
    delay: int = 1,
    default_return: Any = None,
    exceptions: Tuple[Exception, ...] = (Exception,),
    verbose: bool = True
):
    """
    Универсальный декоратор для повторных попыток при ошибках
    
    Args:
        max_retries: Максимальное количество попыток
        delay: Базовая задержка между попытками (секунды)
        default_return: Что возвращать при исчерпании попыток
        exceptions: Какие исключения перехватывать
        verbose: Выводить ли информацию о попытках
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if verbose:
                        print(f"❌ {func.__name__}: Попытка {attempt + 1}/{max_retries} не удалась: {e}")
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Экспоненциальный backoff
                        if verbose:
                            print(f"⏳ Ждем {wait_time} секунд...")
                        time.sleep(wait_time)
                    else:
                        if verbose:
                            print(f"❌ {func.__name__}: Все попытки не удались.")
                        return default_return
            return default_return
        return wrapper
    return decorator