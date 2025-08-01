from datetime import datetime 

def dt_now():
    now = str(datetime.now().replace(second=0, microsecond=0))[:-3].replace(" ", "_").replace(":", "-")
    return now