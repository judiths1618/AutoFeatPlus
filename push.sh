echo ".DS_Store
data/
datasets/
*.zip
*.tar
*.tar.gz
*.csv
*.mat
*.pt
*.pth
*.pkl
*.h5
*.npy
*.npz
" >> .gitignore

git add .gitignore
git commit -m "Add gitignore for large data files"