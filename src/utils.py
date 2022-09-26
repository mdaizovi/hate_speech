from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        base_str = f'Function {func.__name__}{args} {kwargs} Took '
        if total_time >= 60:
            if total_time >= 60*60:
                hours, mins = divmod(total_time, 60*60)
                full_str = base_str + f'{int(hours)} hours {int(mins)} mins'
            else:
                mins, secs = divmod(total_time, 60)
                full_str = base_str + f'{int(mins)} mins {int(secs)} seconds'
        else:
            full_str = base_str + f'{total_time:.2f} seconds'

        print(full_str)
        return result

    return timeit_wrapper
