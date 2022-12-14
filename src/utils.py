import csv
from datetime import datetime
from functools import wraps
import pytz
import time

tz = pytz.timezone('Europe/Berlin')


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        now = datetime.now(tz).strftime("%H:%M:%S")
        try:
            print(f"\nStarting {func.__name__} at {now}")
        except:
            print(f"\nStarting at {now}")

        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        base_str = f'\n*-Function {func.__name__}\n{args}\n{kwargs}\nTook '
        if total_time >= 60:
            if total_time >= 60*60:
                hours, mins = divmod(total_time, 60*60)
                full_str = base_str + f'{int(hours)} hours {int(mins)} mins'
            else:
                mins, secs = divmod(total_time, 60)
                full_str = base_str + f'{int(mins)} mins {int(secs)} seconds'
        else:
            full_str = base_str + f'{total_time:.2f} seconds-*\n\n'

        print(full_str)
        return result

    return timeit_wrapper


def replace_contents(filename='input.csv'):
    """
    Whoops I accidentally output the comments as a list instead of single string
    """
    with open(filename) as csvfile:
        r = csv.reader(csvfile)
        rows_replaced = []
        i = 0
        for row in r:
            if i == 0:
                rows_replaced.append(row)
            else:
                content = row[1]
                denested = content[1:-1]
                rows_replaced.append([row[0], denested, *row[2:]])
            i += 1

    with open('output.csv', 'w') as csvfile:
        w = csv.writer(csvfile)
        for row in rows_replaced:
            w.writerow(row)
