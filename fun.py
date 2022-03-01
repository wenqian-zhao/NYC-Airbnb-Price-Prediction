import winsound
def ring():
    for i, j in zip([415, 330, 370, 247, 247, 370, 415, 330], [800, 800,800,1200,800,800,800,1200]):
        winsound.Beep(i, j)
from datetime import timedelta


def process_in(counter, total, start, end):
    period = end - start
    remain = total - counter
    percent_remain = str(round(100*counter/total, 3))
    print("****"*counter + "----"*remain+"|"+percent_remain\
        + "%       " + str(timedelta(seconds = period)) + "|" + str(timedelta(seconds = period*remain)))