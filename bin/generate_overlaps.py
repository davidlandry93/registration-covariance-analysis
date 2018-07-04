import concurrent

from recova.registration_result_database import RegistrationPairDatabase

db = RegistrationPairDatabase('/home/dlandry/dataset/db_eth_large/')
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
    pairs = db.registration_pairs()
    futures = []
    for pair in pairs:
        f = pool.submit(pair.overlap)
        futures.append(f)

    for i, future in enumerate(futures):
        print(pairs[i])
        print('{} on {}'.format(i, len(futures)))
        print(future.result())
