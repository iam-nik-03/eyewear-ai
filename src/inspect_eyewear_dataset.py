from huggingface_hub import list_repo_files

REPO_ID = "shravya11/eyeglasses-dataset"

print("Inspecting dataset repository...")
print()

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("Total repository files:", len(files))

print("\nFirst 50 files:")
for file in files[:50]:
    print(file)