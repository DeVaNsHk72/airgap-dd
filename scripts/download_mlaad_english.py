from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("mueller91/MLAAD", repo_type="dataset")
langs = sorted({f.split("/")[1] for f in files if f.startswith("fake/") and "/" in f[5:]})
print(langs)