import time
def FUNCTION():
    max_time_end = time.time() + (10)

    cnt = 0
    while True:
        print(cnt)
        cnt += 1

        if time.time() > max_time_end:
            break

FUNCTION()
