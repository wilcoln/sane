def fmt_sizeof(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def fmt_stats_dict(stats_dict, include_time=False):
    if not include_time:
        stats_dict = {k: v for k, v in stats_dict.items() if not k.endswith('_time')}

    return ', '.join([f'{capitalize(k)}: {v:.4f}' for k, v in stats_dict.items()])


def capitalize(underscore_string):
    return ' '.join(w.capitalize() for w in underscore_string.split('_'))
