import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    for slug in ['shuvoalok/raf-db-dataset', 'yakhyokhuja/affectnetaligned', 'vlntnstarodub/datasetsfew']:
        res = api.dataset_list(search=slug)
        for r in res:
            if r.ref.lower() == slug.lower():
                print(f"{r.ref} -> total_bytes: {r.total_bytes / (1024**2):.1f} MB, title: {r.title}")

if __name__ == '__main__':
    main()
