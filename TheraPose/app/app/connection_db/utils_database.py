from datetime import datetime
    
def get_date_today():
    now=datetime.now()
    fmt="%Y-%m-%d %H:%M:%S"
    return now.strftime(fmt)

def diff_days(date):
    now=datetime.now()
    fmt="%Y-%m-%d"
    d1=datetime.strptime(now.strftime(fmt), fmt)
    d2=datetime.strptime(date, fmt)
    return (d1-d2).days